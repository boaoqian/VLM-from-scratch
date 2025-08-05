from model.ViT import ViT

model_path = "/media/qba/Data/Project/DeepLearning/Model/siglip-base-patch16-224"
vit = ViT.from_pretrained(model_path)
print(vit)
