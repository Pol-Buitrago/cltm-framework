from typing import Optional

import numpy as np
import torch
import torch.nn as nn

class QuantizedClassifier(nn.Module):
    """
    Simple token-based classifier:
     - Embedding(num_clusters + 1, dim) where last index is padding embedding
     - Mean pooling ignoring padding
     - Linear head (dim -> num_labels)
    """
    def __init__(self, num_clusters: int, dim: int, num_labels: int, centroids: Optional[np.ndarray] = None,
                 embedding_trainable: bool = False):
        super().__init__()
        self.pad_id = num_clusters  # use last id as padding
        self.embedding = nn.Embedding(num_clusters + 1, dim, padding_idx=self.pad_id)
        if centroids is not None:
            if centroids.shape != (num_clusters, dim):
                raise ValueError("centroids shape mismatch")
            # initialize embeddings[0:num_clusters] with centroids
            with torch.no_grad():
                self.embedding.weight.data[:num_clusters, :] = torch.from_numpy(centroids)
                # initialize pad embedding to zeros
                self.embedding.weight.data[self.pad_id, :].zero_()
        # set trainability
        self.embedding.weight.requires_grad = bool(embedding_trainable)
        self.head = nn.Linear(dim, num_labels)

    def forward(self, input_ids: torch.LongTensor = None, labels: Optional[torch.LongTensor] = None):
        """
        input_ids: LongTensor (batch, seq_len) with values in [0, num_clusters-1] and pad_id for padding
        returns: dict with logits and optionally loss if labels provided
        """
        assert input_ids is not None, "input_ids required for QuantizedClassifier"
        mask = (input_ids != self.pad_id).to(dtype=torch.float32)  # (B, T)
        lengths = mask.sum(dim=1).clamp(min=1.0)  # avoid division by zero
        emb = self.embedding(input_ids)  # (B, T, D)
        # masked sum and mean
        emb_sum = (emb * mask.unsqueeze(-1)).sum(dim=1)  # (B, D)
        pooled = emb_sum / lengths.unsqueeze(-1)  # (B, D)
        logits = self.head(pooled)  # (B, num_labels)
        out = {"logits": logits}
        if labels is not None:
            loss_f = nn.CrossEntropyLoss()
            loss = loss_f(logits, labels)
            out["loss"] = loss
        return out
