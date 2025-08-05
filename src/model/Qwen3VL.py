import torch
import torch.nn as nn
import torch.nn.functional as F
from .Qwen3 import Qwen3ModelForCausalLM
from .ViT import ViT
from typing import Optional, List
from einops import rearrange


class Qwen3VL(nn.Module):
    def __init__(self, config):
        super(Qwen3VL, self).__init__()
        self.config = config
        self.lm = Qwen3ModelForCausalLM(config.llm)
        self.vit = ViT(config.vision)
        self.modality_projector = ModalityProjector(config)
        self.vision_token = config.vision_token
        # todo: mlp

    def forward(self,
                input_ids: torch.LongTensor,
                images: List,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[List] = None,
                is_prefill: bool = False,
                use_cache: Optional[bool] = False):
        # imgs: [batch, imgs]
        imgs = torch.stack(images)
        embs = self.vit(imgs)
        embs = self.modality_projector(embs)
        input_embs = self.get_input_embeddings(input_ids, embs)
        logits, kv_cache = self.lm(inputs_embeds=input_embs,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          kv_cache=kv_cache,
                          is_prefill=is_prefill,
                          use_cache=use_cache)
        if labels is not None:
            logits = logits.view(-1, self.config.llm.vocab_size)
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
            return logits, loss
        return logits, kv_cache
        

    def get_input_embeddings(self, input_ids,
                             img_embs):
        token_embs = self.lm.get_input_embeddings(input_ids)
        mask = input_ids == self.vision_token
        token_embs[mask] = img_embs
        return token_embs
    
    @classmethod
    def from_pretrained(cls, model_path, weight_dict=None):
        pass


class ModalityProjector(nn.Module):
    def __init__(self, config):
        super(ModalityProjector, self).__init__()
        self.config = config
        self.projector = nn.Linear(
            config.vision.hidden_dim*(config.r)**2, config.llm.hidden_size)
        self.w = config.vision.patch_size**0.5
        self.r = config.r  # Pixel shuffle factor
        self.w = int(self.w / self.r)

    def forward(self, embs):
        embs = rearrange(embs, 'b (h r w r) d -> b (h w) (d r r)',
                         h=self.w, w=self.w, r=self.r)
        emb = self.projector(emb)
        return emb
