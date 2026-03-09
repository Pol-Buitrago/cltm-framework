# utils/collators.py
from dataclasses import dataclass
from typing import List, Dict, Union
import torch
from transformers import AutoProcessor

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    features_key_name: str
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{self.features_key_name: feature[self.features_key_name][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
