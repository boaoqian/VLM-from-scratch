import torch.nn as nn
from os import path
from safetensors.torch import load_file
from .Config import Config

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def load_weights(self, weight_dict: dict):
        pass
    
    @classmethod
    def from_pretrained(cls, model_path):
        config = cls.get_model_config(model_path)
        model = cls(config)
        weight_dict = load_file(path.join(model_path, "model.safetensors"))
        model.load_weights(weight_dict)
        return model

    @classmethod
    def get_model_config(cls, model_path) -> Config:
        return Config()