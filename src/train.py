import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from data.dataset import VQADataset
from model.Qwen3VL import Qwen3VL
from model.Config import Qwen3VLConfig
from data.collector import Collector
from safetensors.torch import load_file
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm


# def init_model():
#     config = Qwen3VLConfig.from_pretrained("/media/qba/Data/Project/DeepLearning/Build VLM from scratch/src/Qwen3VL")
#     model = Qwen3VL(config)
#     vit_w = load_file("/media/qba/Data/Project/DeepLearning/Model/siglip-base-patch16-224/model.safetensors")
#     model.vision_model.load_weights(vit_w)
#     lm_w = load_file("/media/qba/Data/Project/DeepLearning/Model/Qwen3-0.6B/model.safetensors")
#     model.lm.load_weights(lm_w)
#     return model

def get_dataloader(data, tokenizer, batch_size=16, device="cuda", dtype="float32"):
    image_processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VQADataset(data, tokenizer, image_processor, 49)
    collector = Collector(tokenizer.pad_token_id, device, dtype)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collector)
    return data_loader

def train(model, tokenizer, data, epoch, batch_size, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)
    writer = SummaryWriter()
    model.train()

    global_step = 0
    loss_sum = 0
    log_freq = 100

    for i in range(epoch):
        data_loader = get_dataloader(data, tokenizer, batch_size)
        for batch in tqdm(data_loader):
            outputs = model(**batch)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item()
            global_step += 1

            if global_step % log_freq == 0:
                # log loss
                writer.add_scalar("train/loss", loss_sum / log_freq, global_step)
                loss_sum = 0

                # log 当前学习率
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/lr", current_lr, global_step)

                # log 权重范数 (L2 norm)
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.data.detach().norm(2)  # L2 范数
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar("train/weight_norm", total_norm, global_step)

                # log 梯度范数 (可选)
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.detach().norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                print("epoch:{i+1} step:{global_step} loss:{loss_sum / log_freq} lr:{current_lr} weight_norm:{total_norm} grad_norm:{grad_norm}")


def main():
    parser = argparse.ArgumentParser(description="training args")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_path", type=str, default="/media/qba/Data/Project/DeepLearning/Dataset/the_cauldron/tqa/train-00000-of-00001-c15be8aed9c93862.parquet")
    parser.add_argument("--model_path", type=str, default="/media/qba/Data/Project/DeepLearning/Build VLM from scratch/src/Qwen3VL")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = Qwen3VL.from_pretrained(args.model_path).to("cuda")
    data = load_dataset("parquet", data_files=args.data_path)["train"]
    train(model, tokenizer, data, args.epoch, args.batch_size)

main()