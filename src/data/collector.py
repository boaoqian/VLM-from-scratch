import torch

'''
        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
'''

class Collector:
    def __init__(self, pad_token_id, device = "cpu", torch_type = torch.bfloat16, max_length=None):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.device = device
        self.torch_type = torch_type

    def __call__(self, examples):
        batch = {"images":[],"input_ids":[],"attention_mask":[],"labels":[]}
        if self.max_length is None:
            self.max_length = max(map(len, [d["input_ids"] for d in examples]))

        for data in examples:
            batch["images"].append(torch.stack(data["images"]).to(self.device).to(self.torch_type))
            batch["input_ids"].append(
                torch.nn.functional.pad(data["input_ids"],(0, self.max_length-len(data["input_ids"])), "constant", self.pad_token_id
                ))
            batch["attention_mask"].append(
                torch.nn.functional.pad(data["attention_mask"],(0, self.max_length-len(data["attention_mask"])), "constant", self.pad_token_id)
                )
            batch["labels"].append(
                torch.nn.functional.pad(data["labels"],(0, self.max_length-len(data["labels"])), "constant", -100)
                )
        
        return {
            "input_ids": torch.stack(batch["input_ids"]).to(self.device),
            "attention_mask": torch.stack(batch["attention_mask"]).to(self.device),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]).to(self.device),
        }

        