"""
Small helpers and collators for continuous and quantized training.
"""
import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

class HFDatasetWrapper(torch.utils.data.Dataset):
    """Wrap an HF dataset split together with precomputed token lists."""

    def __init__(self, hf_dataset_split: Any, token_lists: Iterable[Iterable[int]]):
        self.hf = hf_dataset_split
        self.token_lists = list(token_lists)
        if hasattr(self.hf, "__len__"):
            assert len(self.hf) == len(self.token_lists), "length mismatch"

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.hf[idx])
        item["input_ids"] = np.asarray(self.token_lists[idx], dtype=np.int64)
        return item

# --- small helpers ---
def _labels_to_tensor(batch: List[Dict[str, Any]], args) -> torch.LongTensor:
    """Convert labels from batch to a torch long tensor using args.label2id."""
    labels = [
        args.label2id[str(x["label"])] if isinstance(x["label"], str) else int(x["label"])
        for x in batch
    ]
    return torch.tensor(labels, dtype=torch.long)

# ---------------------
# Continuous collator
# Short: pad raw input_values and return labels.
# ---------------------
def collate_fn_cont_factory(args):
    def collate_fn_cont(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_tensors = [torch.tensor(x["input_values"], dtype=torch.float32) for x in batch]
        input_values = pad_sequence(input_tensors, batch_first=True)
        labels = _labels_to_tensor(batch, args)
        return {"input_values": input_values, "labels": labels}

    return collate_fn_cont

# ---------------------
# Quantized collator
# Short: pad precomputed token sequences and return labels.
# ---------------------
def make_collate_fn_quant(feature_extractor, model_hidden, centroids, args, device):
    pad_id = int(args.n_clusters)

    def seq_to_tensor(seq) -> torch.LongTensor:
        """Convert seq-like to contiguous long tensor."""
        if isinstance(seq, torch.Tensor):
            return seq.long().contiguous()
        if isinstance(seq, np.ndarray):
            return torch.from_numpy(seq).long().contiguous()
        return torch.tensor(list(seq), dtype=torch.long)

    def collate_fn_quant(batch):
        """Simple collator for quantized mode using precomputed token IDs."""
        first = batch[0] if len(batch) else {}
        name = "input_ids" if "input_ids" in first else "token_ids"

        seqs = [seq_to_tensor(x[name]) for x in batch]
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        labels = torch.tensor(
            [args.label2id[str(x["label"])] if isinstance(x["label"], str) else int(x["label"]) for x in batch],
            dtype=torch.long,
        )
        return {"input_ids": padded, "labels": labels}

    return collate_fn_quant
