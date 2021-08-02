from pathlib import Path
from typing import Callable

import torch
from src.data_module import DataModule
from src.model import DefSent
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


@torch.no_grad()
def get_mrr(indices, targets):
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero(as_tuple=False)
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    return torch.sum(rranks)


class Def2WordEvaluation:
    def __init__(
        self,
        data_module: DataModule,
        tokenizer: PreTrainedTokenizerBase,
        topk: int = 10,
        save_predictions: bool = False,
    ) -> None:
        self.dm = data_module
        self.tokenizer = tokenizer
        self.topk = topk
        self.save_predictions = save_predictions

    @torch.no_grad()
    def __call__(self, model: DefSent, mode: str):
        if mode == "train":
            dataset = self.dm.train
            dataloader = self.dm.train_dataloader()
        elif mode == "val":
            dataset = self.dm.val
            dataloader = self.dm.val_dataloader()
        elif mode == "test":
            dataset = self.dm.test
            dataloader = self.dm.test_dataloader()
        else:
            raise ValueError(f"No such a mode!: {mode}")

        res = []
        mrr_sum = 0
        topk_acc_sum = [0] * self.topk
        device = model.device

        for batch in tqdm(dataloader):
            words_ids, definitions_ids, attention_mask = batch
            words_ids, definitions_ids, attention_mask = (
                words_ids.to(device),
                definitions_ids.to(device),
                attention_mask.to(device),
            )

            logits = model.predict_words(definitions_ids, attention_mask=attention_mask)
            hypothesis = logits.topk(self.topk, dim=1).indices
            words = self.tokenizer.convert_ids_to_tokens(words_ids)

            for word, definition_ids, hyp_words_ids in zip(
                words, definitions_ids, hypothesis
            ):
                hyp_words = self.tokenizer.convert_ids_to_tokens(hyp_words_ids)
                assert len(hyp_words) == self.topk

                if self.save_predictions:
                    definition_tokens = self.tokenizer.convert_ids_to_tokens(
                        definition_ids, skip_special_tokens=True
                    )
                    definition = self.tokenizer.convert_tokens_to_string(
                        definition_tokens
                    )
                    res.append(
                        {"word": word, "definition": definition, "hyp_words": hyp_words}
                    )

                already_found_correct_word = False
                for i in range(self.topk):
                    if hyp_words[i] == word:
                        already_found_correct_word = True
                    if already_found_correct_word:
                        topk_acc_sum[i] += 1

            mrr_sum += get_mrr(hypothesis, words_ids).item()

        ret = {
            mode: {
                "MRR": mrr_sum / len(dataset) * 100,
                "ACC": [cnt / len(dataset) * 100 for cnt in topk_acc_sum],
            }
        }
        if self.save_predictions:
            ret[mode]["result"] = res
        return ret


class Def2WordEvaluationAll:
    def __init__(
        self,
        data_module: DataModule,
        tokenizer: PreTrainedTokenizerBase,
        topk: int = 10,
        save_predictions: bool = False,
        log_artifact: Callable[[str], None] = None,
    ) -> None:
        self.save_predictions = save_predictions
        self.def2word_evaluator = Def2WordEvaluation(
            data_module=data_module,
            tokenizer=tokenizer,
            topk=topk,
            save_predictions=save_predictions,
        )
        self.log_artifact = log_artifact

    def __call__(self, model: DefSent):
        if self.save_predictions:
            results_dir = Path("./results/def2word-prediction")
            results_dir.mkdir(parents=True, exist_ok=True)

        metrics = {}
        for mode in ["train", "val", "test"]:
            result = self.def2word_evaluator(model, mode=mode)
            topk_acc = result[mode]["ACC"]
            top1, top3, top10 = topk_acc[0], topk_acc[2], topk_acc[9]
            mrr = result[mode]["MRR"]
            metrics[mode] = {"MRR": mrr, "top1": top1, "top3": top3, "top10": top10}

            if self.save_predictions:
                save_path = results_dir / f"{mode}.txt"
                res = result[mode]["result"]
                lines = []
                for data in res:
                    word, definition, hyp_words = (
                        data["word"],
                        data["definition"],
                        data["hyp_words"],
                    )
                    hyp_line = "\t".join(hyp_words)
                    lines.append(f"{word}\t[{definition}]\n{hyp_line}\n")
                save_path.write_text("\n".join(lines))
                self.log_artifact(save_path)

        return metrics
