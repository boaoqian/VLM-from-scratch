import torch
import torch.nn as nn
import torch.nn.functional as F
from .Qwen3 import Qwen3ModelForCausalLM
from .ViT import ViT
from typing import Optional, List
from einops import rearrange
from .BaseModel import BaseModel
from .Config import Qwen3VLConfig

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.detach())

class Qwen3VL(BaseModel):
    def __init__(self, config):
        super(Qwen3VL, self).__init__()
        self.config = config
        self.lm = Qwen3ModelForCausalLM(config.llm)
        self.vit = ViT(config.vision)
        self.modality_projector = ModalityProjector(config)
        self.vision_token_id = config.vision_token_id

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
        if isinstance(images, List):
            imgs = torch.concatenate([img_list for img_list in images])
        embs = self.vit(imgs)
        embs = self.modality_projector(embs)
        input_embs = self.get_input_embeddings(input_ids, embs)
        logits, kv_cache = self.lm(inputs_embeds=input_embs,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          kv_cache=kv_cache,
                          is_prefill=is_prefill)
        if labels is not None:
            logits = logits.view(-1, self.config.llm.vocab_size)
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
            return {"logits":logits, "loss":loss}
        return {"logits":logits, "kv_cache":kv_cache}
        

    def get_input_embeddings(self, input_ids,
                             img_embs):
        token_embs = self.lm.get_input_embeddings(input_ids)
        mask = (input_ids == self.vision_token_id)
        token_embs[mask] = img_embs.view(-1, img_embs.shape[-1])
        return token_embs
    
    @classmethod
    def get_model_config(cls, model_path):
        return Qwen3VLConfig.from_pretrained(model_path)
    
    def load_weights(self, weight_dict):
        lm_weight = {k[3:]: v for k, v in weight_dict.items() if k.startswith("lm.")}
        vit_weight = {k[4:]: v for k, v in weight_dict.items() if k.startswith("vit.")}
        self.lm.load_weights(lm_weight)
        self.vit.load_weights(vit_weight)
        self.modality_projector.projector.weight = assign(self.modality_projector.projector.weight, weight_dict["modality_projector.projector.weight"])
        self.modality_projector.projector.bias = assign(self.modality_projector.projector.bias, weight_dict["modality_projector.projector.bias"])
        


class ModalityProjector(nn.Module):
    def __init__(self, config):
        super(ModalityProjector, self).__init__()
        self.config = config
        self.projector = nn.Linear(
            config.vision.hidden_size*(config.r)**2, config.llm.hidden_size)
        self.w = config.vision.image_size//config.vision.patch_size//config.r
        self.r = config.r  # Pixel shuffle factor

    def forward(self, embs):
        embs = rearrange(embs, 'b (h r1 w r2) d -> b (h w) (d r1 r2)',
                 h=self.w, w=self.w, r1=self.r, r2=self.r)

        embs = self.projector(embs)
        return embs
