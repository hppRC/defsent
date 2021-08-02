from typing import List

import torch.nn as nn
from torch import Tensor


def pad_sequence(
    sequences: List[Tensor], padding_value: int, padding_side: str = "right"
):
    if padding_side == "right":
        return right_side_padding(sequences, padding_value)
    elif padding_side == "left":
        return left_side_padding(sequences, padding_value)
    else:
        raise ValueError(f"no such a padding side name! > {padding_side}")


def right_side_padding(sequences: List[Tensor], padding_value: int):
    return nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=padding_value,
    )


def left_side_padding(sequences: List[Tensor], padding_value: int):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        # use index notation to prevent duplicate references to the tensor
        length = tensor.size(0)
        out_tensor[i, -length:, ...] = tensor

    return out_tensor
