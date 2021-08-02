from pathlib import Path
from typing import Callable, List, Tuple, Union

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        text_to_data: Callable[[List[str]], Tuple[List[int], List[int]]],
    ):
        with Path(data_path).open() as f:
            self.words, self.definitions = text_to_data(f.readlines())

        assert len(self.words) == len(self.definitions)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            return (
                torch.LongTensor(self.words[key]),
                torch.LongTensor(self.definitions[key]),
            )
        elif isinstance(key, slice):
            return (
                [torch.LongTensor(word_id) for word_id in self.words[key]],
                [
                    torch.LongTensor(definition_ids)
                    for definition_ids in self.definitions[key]
                ],
            )
