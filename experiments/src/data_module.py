import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from src.dataset import Dataset
from src.utils import pad_sequence
from torch.functional import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir: Union[Path, str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer

        self.train = None
        self.val = None
        self.test = None

    def text_to_data(self, lines: List[str]) -> Tuple[List[int], List[int]]:
        words, definitions = [], []
        for line in lines:
            word, definition = line.strip().split("\t")  # line: "word\tdefinition"
            words.append(word)
            definitions.append(definition)

        # encode without special tokens (e.g., [CLS], [SEP], <s>, <\s>)
        words_ids = self.tokenizer(words, add_special_tokens=False).input_ids
        definitions_ids = self.tokenizer(definitions, truncation=True).input_ids

        filtered_words_ids, filtered_definitions_ids = [], []
        for word_id, definition_ids in zip(words_ids, definitions_ids):
            if len(word_id) == 1:
                filtered_words_ids.append(word_id)
                filtered_definitions_ids.append(definition_ids)

        return (filtered_words_ids, filtered_definitions_ids)

    def collate_fn(
        self, data_list: List[Tuple[List[Tensor], List[Tensor]]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        word_id_list, definition_ids_list = zip(*data_list)
        words_ids = torch.cat(word_id_list, dim=0)
        definitions_ids = pad_sequence(
            definition_ids_list,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
        )
        attention_mask = (definitions_ids != self.tokenizer.pad_token_id).float()

        return (words_ids, definitions_ids, attention_mask)

    def setup(self, stage: Optional[str] = None) -> None:
        # make assignments here (train/valid/test split)
        # called on every GPUs
        self.train = Dataset(
            data_path=self.data_dir / "train.tsv", text_to_data=self.text_to_data,
        )
        self.val = Dataset(
            data_path=self.data_dir / "valid.tsv", text_to_data=self.text_to_data,
        )
        self.test = Dataset(
            data_path=self.data_dir / "test.tsv", text_to_data=self.text_to_data,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
            # pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
            # pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
            # pin_memory=True,
        )
