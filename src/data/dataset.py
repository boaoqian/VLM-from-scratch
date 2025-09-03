import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.image_token = "<|vision_start|>"
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string = "qba111"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _get_messages(self, item, image_count=0):
        messages = []
        for text in item['texts']:
            messages.append({"role": "user", "content": text['user']})
            messages.append({"role": "assistant", "content": text['assistant']})

        if image_count > 0:
            messages[0]["content"] = self.image_token * image_count * self.mp_image_token_length + messages[0]["content"]      

        return messages

    def _process_images(self, images):
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image = self.image_processor(image)
                processed_images.append(processed_image)
            else:
                raise ValueError("Error processing image")
        return processed_images


    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end   = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return torch.tensor(conv_ids["input_ids"]), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle images (should be a list)
        images_data = item['images']
        if not isinstance(images_data, list):
            images_data = [images_data]

        # Now process the images
        processed_images = self._process_images(images_data)

        messages = self._get_messages(item, len(processed_images))

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1) # Shift labels for causal LM
        labels[-1] = -100 # Last token has no target
        
        return labels
    
if __name__ == "__main__":
    from datasets import load_dataset
    from torchvision import transforms
    from transformers import AutoTokenizer

    data = load_dataset("parquet", data_files="/media/qba/Data/Project/DeepLearning/Dataset/the_cauldron/tqa/train-00000-of-00001-c15be8aed9c93862.parquet")["train"]
    tokenizer = AutoTokenizer.from_pretrained("/media/qba/Data/Project/DeepLearning/Model/Qwen3-0.6B")
    image_processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VQADataset(data, tokenizer, image_processor, 64)
    print(dataset[0])

    
