import json
import os
from typing import Optional


class Config:
    def __init__(self,
                 pad_token_id=1,
                 bos_token_id=49406,
                 eos_token_id=49407,
                 **kwargs):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    def from_pretrained(cls, model_path):
        cfg = json.load(open(os.path.join(model_path, "config.json")))
        return cls(**cfg)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)


class ViTConfig(Config):
    def __init__(
        self,
            hidden_size: int = 768,
            intermediate_size: int = 3072,
            patch_size: int = 16,
            image_size: int = 224,
            num_attention_heads: int = 12,
            attention_dropout: float = 0.0,
            num_hidden_layers: int = 12,
            layer_norm_eps: float = 1e-6,
            vit_cls_flag: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.patch_size = patch_size
        self.image_size = image_size
        self.vit_cls_flag = vit_cls_flag

    @classmethod
    def from_pretrained(cls, model_path):
        cfg = json.load(open(os.path.join(model_path, "config.json")))
        if cfg.get("vision_config", None) is None:
            return cls(**cfg)
        else:
            return cls(**cfg["vision_config"])


class Qwen3Config(Config):
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


class Qwen3VLConfig(Config):
    def __init__(
        self,
        r: int = 2,
        vision_token: str = "<|image_pad|>",
        llm_config: Optional[dict] = {},
        vision_config: Optional[dict] = {},
        **kwargs
    ):
        self.r = r
        self.vision_token = vision_token
        self.llm = Qwen3Config(**llm_config)
        self.vision = ViTConfig(**vision_config)
        super().__init__(**kwargs)




