from safetensors.torch import load_file

path = "/media/qba/Data/Project/DeepLearning/Model/siglip-base-patch16-224/model.safetensors"
weight_dict = load_file(path)
for item in weight_dict:
    print(item)