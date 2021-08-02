import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Union
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from defsent.pooling import Pooling


class DefSent(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device = None,
    ) -> None:
        super(DefSent, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.pooling_name = model_name_or_path.rsplit("-", 1)[-1]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoder, self.prediction_layer = pretrained_modules(model_name_or_path)
        self.pooling = Pooling(pooling_name=self.pooling_name)

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.to(self.device)

    def to(self, device: torch.device) -> None:
        self.encoder = self.encoder.to(device)
        self.prediction_layer = self.prediction_layer.to(device)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        embs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        emb = self.pooling(embs, attention_mask=attention_mask)
        return emb

    def calc_word_logits(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        emb = self(input_ids, attention_mask=attention_mask)
        logits = self.prediction_layer(emb)
        return logits

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 16,
    ) -> Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        data_loader = torch.utils.data.DataLoader(
            list(zip(inputs.input_ids, inputs.attention_mask)),
            batch_size=batch_size,
        )
        all_embs = []
        for input_ids, attention_mask in data_loader:
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            embs = self.forward(input_ids, attention_mask=attention_mask)
            # Prevent overuse of memory.
            embs = embs.cpu()
            all_embs.append(embs)

        embeddings = torch.cat(all_embs, dim=0)
        return embeddings

    @torch.no_grad()
    def predict_words(
        self,
        sentences: Union[str, List[str]],
        topk: int = 10,
        batch_size: int = 16,
    ) -> List[List[str]]:
        embs = self.encode(
            sentences=sentences,
            batch_size=batch_size,
        )
        logits: Tensor = self.prediction_layer(embs.to(self.device)).cpu()
        hypothesis = logits.topk(topk, dim=1).indices
        words = [self.tokenizer.convert_ids_to_tokens(hyp_ids) for hyp_ids in hypothesis]
        return words


def pretrained_modules(model_name_or_path: str) -> Tuple[nn.Module, nn.Module]:
    config = AutoConfig.from_pretrained(model_name_or_path)

    if "BertForMaskedLM" in config.architectures:
        pretrained_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        encoder = pretrained_model.bert
        prediction_layer = pretrained_model.cls

    elif "RobertaForMaskedLM" in config.architectures:
        pretrained_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        encoder = pretrained_model.roberta
        prediction_layer = pretrained_model.lm_head

    else:
        raise ValueError(f"No such a pre-trained model! > {model_name_or_path}")

    return encoder, prediction_layer