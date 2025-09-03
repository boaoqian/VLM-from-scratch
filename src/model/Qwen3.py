import torch
import torch.nn as nn
from typing import Optional, List
from .Config import Qwen3Config
from .BaseModel import BaseModel


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(
            self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(
            module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # unlike olmo, only on the head dim!
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # thus post q_norm does not need reshape
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_cache: Optional[dict] = None,
        is_prefill: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(
            hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(
            hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
        
        if kv_cache is not None and not is_prefill:
            self.is_causal = False
            key_states = torch.cat([kv_cache["key"], key_states], dim=2)
            value_states = torch.cat([kv_cache["value"], value_states], dim=2)
            kv_cache["key"] = key_states
            kv_cache["value"] = value_states
            attention_mask = torch.ones(query_states.shape[2],key_states.shape[2], device=query_states.device, dtype=torch.bool)
        elif is_prefill:
            self.is_causal = True
            kv_cache = {
                "key": key_states,
                "value": value_states,
            }
        key = repeat_kv(key_states, self.num_key_value_groups)
        value = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]

        # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        query = query_states.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        if self.is_causal:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attention_dropout,
                scale=self.scaling,
                is_causal=self.is_causal,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout,
                scale=self.scaling,
                is_causal=self.is_causal,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        # if self.layer_idx==0:
        #     import pickle,time
        #     pickle.dump((query, key, value, attn_output), open(str(time.time())+str(key.shape)+"qkvo.pkl", "wb"))

        return attn_output, kv_cache



class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            kv_cache: Optional[dict] = None,
            is_prefill: bool = True,
            ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            kv_cache = kv_cache,
            is_prefill=is_prefill,

        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, kv_cache


def _compute_default_rope_parameters(
    config=None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2,
                      dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(
                x.shape[-2], device=x.device, dtype=torch.long).repeat(x.shape[0], 1)
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(
            x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @
                     position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.detach())


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    '''
    model.layers.0.input_layernorm.weight torch.Size([1024])
    model.layers.0.mlp.down_proj.weight torch.Size([1024, 3072])
    model.layers.0.mlp.gate_proj.weight torch.Size([3072, 1024])
    model.layers.0.mlp.up_proj.weight torch.Size([3072, 1024])
    model.layers.0.post_attention_layernorm.weight torch.Size([1024])
    model.layers.0.self_attn.k_norm.weight torch.Size([128])
    model.layers.0.self_attn.k_proj.weight torch.Size([1024, 1024])
    model.layers.0.self_attn.o_proj.weight torch.Size([1024, 2048])
    model.layers.0.self_attn.q_norm.weight torch.Size([128])
    model.layers.0.self_attn.q_proj.weight torch.Size([2048, 1024])
    model.layers.0.self_attn.v_proj.weight torch.Size([1024, 1024])
    '''

    def load_weights(self, weight_dict):
        self.embed_tokens.weight = assign(
            self.embed_tokens.weight, weight_dict["model.embed_tokens.weight"])
        self.norm.weight = assign(
            self.norm.weight, weight_dict["model.norm.weight"])
        for i, layer in enumerate(self.layers):
            layer.input_layernorm.weight = assign(
                layer.input_layernorm.weight, weight_dict[f"model.layers.{i}.input_layernorm.weight"])
            layer.post_attention_layernorm.weight = assign(
                layer.post_attention_layernorm.weight, weight_dict[f"model.layers.{i}.post_attention_layernorm.weight"])
            layer.self_attn.q_norm.weight = assign(
                layer.self_attn.q_norm.weight, weight_dict[f"model.layers.{i}.self_attn.q_norm.weight"])
            layer.self_attn.k_norm.weight = assign(
                layer.self_attn.k_norm.weight, weight_dict[f"model.layers.{i}.self_attn.k_norm.weight"])
            layer.self_attn.q_proj.weight = assign(
                layer.self_attn.q_proj.weight, weight_dict[f"model.layers.{i}.self_attn.q_proj.weight"])
            layer.self_attn.k_proj.weight = assign(
                layer.self_attn.k_proj.weight, weight_dict[f"model.layers.{i}.self_attn.k_proj.weight"])
            layer.self_attn.v_proj.weight = assign(
                layer.self_attn.v_proj.weight, weight_dict[f"model.layers.{i}.self_attn.v_proj.weight"])
            layer.self_attn.o_proj.weight = assign(
                layer.self_attn.o_proj.weight, weight_dict[f"model.layers.{i}.self_attn.o_proj.weight"])
            layer.mlp.down_proj.weight = assign(
                layer.mlp.down_proj.weight, weight_dict[f"model.layers.{i}.mlp.down_proj.weight"])
            layer.mlp.gate_proj.weight = assign(
                layer.mlp.gate_proj.weight, weight_dict[f"model.layers.{i}.mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight = assign(
                layer.mlp.up_proj.weight, weight_dict[f"model.layers.{i}.mlp.up_proj.weight"])

    @classmethod
    def get_model_config(cls, model_path):
        return Qwen3Config.from_pretrained(model_path)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[List] = None,
        is_prefill: bool = True,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask == 1

        hidden_states = inputs_embeds


        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if kv_cache is None:   
            kv_cache = [None] * len(self.layers) 

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_outputs, kv_cache[i]= decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                is_prefill=is_prefill,
                kv_cache=kv_cache[i],
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        return hidden_states, kv_cache


class Qwen3ModelForCausalLM(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[List] = None,
            is_prefill: bool = True):
        
        hidden_states, kv_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
            is_prefill=is_prefill,
        )
        logits = self.lm_head(hidden_states)
        return logits, kv_cache

    def get_input_embeddings(self,input_ids):
        return self.model.embed_tokens(input_ids)

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    @classmethod
    def get_model_config(cls, model_path):
        return Qwen3Config.from_pretrained(model_path)
    
    def load_weights(self, weight_dict):
        self.model.load_weights(weight_dict)
        self.lm_head.weight = assign(
            self.lm_head.weight, weight_dict["lm_head.weight"])

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 10,
        stop_tokens: Optional[List[int]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        kv_cache = None
        if inputs_embeds is None and input_ids is None:
            raise ValueError(
                "You must specify either input_ids or inputs_embeds for generation"
            )
        if input_ids is not None:
            output_ids = [[]]*input_ids.shape[0]
        else:
            output_ids = [[]]*inputs_embeds.shape[0]
        for step in range(max_new_tokens):
            if use_cache:
                if step == 0:
                    logits, kv_cache = self.forward(
                        input_ids=input_ids,
                        inputs_embeds = inputs_embeds,
                    )
                else:
                    logits, kv_cache = self.forward(
                        input_ids=next_token,
                        inputs_embeds = inputs_embeds,
                        kv_cache = kv_cache,
                        is_prefill = False,
                        position_ids=torch.tensor([[input_ids.shape[1]-1]], device=input_ids.device) # Update position_ids for next token
                    )
            else:
                logits, kv_cache = self.forward(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                )
            next_token_logits = logits[..., -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if input_ids is None:
                input_ids = next_token
            else:
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            inputs_embeds = None

            # 检查是否是停止词
            if stop_tokens:
                for i in range(input_ids.shape[0]):
                    if next_token[i].item() in stop_tokens and output_ids[i] == []:
                        output_ids[i] = input_ids[i, :-1].tolist()
                if all(row for row in output_ids):
                    break


        for i in range(len(output_ids)):
            if output_ids[i] == []:
                output_ids[i] = input_ids[i].tolist()
        return output_ids,kv_cache
