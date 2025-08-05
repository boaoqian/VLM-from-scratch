## 一个vlm需要什么样的模型架构
The vision and text modalities are aligned using a **Modality Projection module**. This module takes the image embeddings produced by the vision backbone as input, and transforms them into embeddings compatible with the text embeddings from the embedding layer of the language model. These embeddings are then concatenated and fed into the language decoder. The Modality Projection module consists of a **pixel shuffle** operation followed by a linear layer.
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nanovlm/architecture.png)

## 具体架构
lm使用qwen3系列
vit 使用siglip-base-patch16-224
