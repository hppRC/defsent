import torch
import torch.nn as nn
from torch import Tensor


class Pooling(nn.Module):
    def __init__(self, pooling_name: str) -> None:
        super().__init__()
        self.pooling_name = pooling_name

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling_name == "cls":
            return x[:, 0]

        if self.pooling_name == "sep":
            # masked tokens are marked as `0`
            sent_len = attention_mask.sum(dim=1, keepdim=True)
            batch_size = x.size(0)
            batch_indices = torch.LongTensor(range(batch_size))
            sep_indices = (sent_len.long() - 1).squeeze()
            return x[batch_indices, sep_indices]

        mask_value = 0 if self.pooling_name in ["mean", "sum"] else -1e6
        x[attention_mask.long() == 0, :] = mask_value

        if self.pooling_name == "mean":
            sent_len = attention_mask.sum(dim=1, keepdim=True)
            return x.sum(dim=1) / sent_len

        elif self.pooling_name == "max":
            return x.max(dim=1).values

        elif self.pooling_name == "sum":
            return x.sum(dim=1)

        else:
            raise ValueError(f"No such a pooling name! {self.pooling_name}")
