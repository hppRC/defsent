from typing import Tuple

import pytorch_lightning as pl
import torch.nn as nn
from src.pooling import NonParametricPooling
from torch import Tensor
from transformers import (
    AlbertForMaskedLM,
    BertForMaskedLM,
    DebertaForMaskedLM,
    PreTrainedModel,
    RobertaForMaskedLM,
)


class DefSent(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pooling_name: str,
        randomize_prediction_layer: bool = False,
        freeze_prediction_layer: bool = True,
        freeze_token_embeddings: bool = True,
    ) -> None:
        super().__init__()
        # When `freeze_prediction_layer or freeze_token_embeddings` is `False`, we should not tie `word_embeddings` and `prediction_layer.decoder`;
        # otherwise, when the parameters of one of them are updated, the other will be updated
        tie_word_embeddings = freeze_prediction_layer and freeze_token_embeddings
        (
            self.pretrained_model,
            self.encoder,
            self.token_embeddings,
            self.prediction_layer,
        ) = pretrained_modules(
            model_name=model_name, tie_word_embeddings=tie_word_embeddings,
        )

        if randomize_prediction_layer:
            nn.init.normal_(self.prediction_layer.weight)
        if freeze_prediction_layer:
            for param in self.prediction_layer.parameters():
                param.requires_grad = False
        if freeze_token_embeddings:
            for param in self.token_embeddings.parameters():
                param.requires_grad = False

        self.pooling = NonParametricPooling(pooling_name=pooling_name)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        embs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        emb = self.pooling(embs, attention_mask=attention_mask)
        return emb

    def predict_words(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        emb = self(input_ids, attention_mask=attention_mask)
        logits = self.prediction_layer(emb)
        return logits


# Each pretrained model have different architecture and name.
# This function performs like an `adapter`.
def pretrained_modules(
    model_name: str, tie_word_embeddings: bool,
) -> Tuple[PreTrainedModel, nn.Module, nn.Module, nn.Module]:
    if model_name in [
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-cased",
        "bert-large-cased",
        "bert-base-multilingual-uncased",
        "bert-base-multilingual-cased",
        "bert-base-chinese",
        "bert-base-german-cased",
        "bert-large-uncased-whole-word-masking",
        "bert-large-cased-whole-word-masking",
        "bert-large-uncased-whole-word-masking-finetuned-squad",
        "bert-large-cased-whole-word-masking-finetuned-squad",
        "bert-base-cased-finetuned-mrpc",
        "bert-base-german-dbmdz-cased",
        "bert-base-german-dbmdz-uncased",
        "cl-tohoku/bert-base-japanese",
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        "cl-tohoku/bert-base-japanese-char",
        "cl-tohoku/bert-base-japanese-char-whole-word-masking",
        "TurkuNLP/bert-base-finnish-cased-v1",
        "TurkuNLP/bert-base-finnish-uncased-v1",
        "wietsedv/bert-base-dutch-cased",
        # See all BERT models at https://huggingface.co/models?filter=bert
    ]:
        pretrained_model = BertForMaskedLM.from_pretrained(
            model_name, tie_word_embeddings=tie_word_embeddings,
        )
        encoder = pretrained_model.bert
        token_embeddings = pretrained_model.bert.embeddings
        prediction_layer = pretrained_model.cls

    elif model_name in [
        "roberta-base",
        "roberta-large",
        "xlm-roberta-base",
        "xlm-roberta-large",
    ]:
        pretrained_model = RobertaForMaskedLM.from_pretrained(
            model_name, tie_word_embeddings=tie_word_embeddings,
        )
        encoder = pretrained_model.roberta
        token_embeddings = pretrained_model.roberta.embeddings
        prediction_layer = pretrained_model.lm_head

    elif model_name in ["albert-base-v2", "albert-large-v2"]:
        pretrained_model = AlbertForMaskedLM.from_pretrained(
            model_name, tie_word_embeddings=tie_word_embeddings,
        )
        encoder = pretrained_model.albert
        token_embeddings = pretrained_model.albert.embeddings
        prediction_layer = pretrained_model.predictions

    elif model_name in [
        "microsoft/deberta-base",
        "microsoft/deberta-large",
        "microsoft/deberta-xlarge",
        "microsoft/deberta-base-mnli",
        "microsoft/deberta-large-mnli",
        "microsoft/deberta-xlarge-mnli",
        "microsoft/deberta-v2-xlarge",
        "microsoft/deberta-v2-xxlarge",
        "microsoft/deberta-v2-xlarge-mnli",
        "microsoft/deberta-v2-xxlarge-mnli",
    ]:
        pretrained_model = DebertaForMaskedLM.from_pretrained(
            model_name, tie_word_embeddings=tie_word_embeddings,
        )
        encoder = pretrained_model.deberta
        token_embeddings = pretrained_model.deberta.embeddings
        prediction_layer = pretrained_model.lm_predictions

    else:
        raise ValueError(f"no such a model name! > {model_name}")

    return pretrained_model, encoder, token_embeddings, prediction_layer
