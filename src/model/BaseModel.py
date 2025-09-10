import torch
import torch.nn as nn
import os
from os import path
from safetensors.torch import load_file, save_file
from .Config import Config

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def load_weights(self, weight_dict: dict):
        self.load_state_dict(weight_dict)

    def save_weights(self, save_path: str):
        """
        保存模型权重，支持 .pt/.pth 和 .safetensors 格式
        Args:
            save_path (str): 权重文件保存路径，例如 'model.safetensors' 或 'model.pth'
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state_dict = self.state_dict()

        # 根据文件后缀判断保存方式
        if save_path.endswith(".safetensors"):
            save_file(state_dict, save_path)
            print(f"[INFO] 模型权重已保存为 SafeTensors 格式: {save_path}")
        elif save_path.endswith(".pt") or save_path.endswith(".pth"):
            torch.save(state_dict, save_path)
            print(f"[INFO] 模型权重已保存为 PyTorch 格式: {save_path}")
        else:
            raise ValueError(
                "Unsupported file extension. Use '.safetensors', '.pt', or '.pth'."
            )
    
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