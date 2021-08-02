import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from src.data_module import DataModule
from src.evaluation import Def2WordEvaluationAll, STSEvaluation
from src.model import DefSent
from torch import Tensor
from torch.optim import Optimizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Experiment(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super(Experiment, self).__init__()
        self.config: DictConfig = config
        logger = instantiate(config.logger)
        self.trainer = instantiate(
            config.trainer,
            logger=logger,
            # callbacks=[LearningRateMonitor(logging_interval="step")],
        )
        self.model: DefSent = instantiate(config.model)
        self.tokenizer: PreTrainedTokenizerBase = instantiate(config.tokenizer)
        self.data_module: DataModule = instantiate(
            config.data_module, tokenizer=self.tokenizer
        )

        self.def2word_evaluator = Def2WordEvaluationAll(
            data_module=self.data_module,
            tokenizer=self.tokenizer,
            topk=config.d2w.topk,
            save_predictions=config.d2w.save_predictions,
            log_artifact=self.log_artifact,
        )
        self.sts_evaluator = STSEvaluation(data_dir=config.sts.data_dir)

    def configure_optimizers(self):
        params = (param for param in self.model.parameters() if param.requires_grad)
        steps_per_epoch = len(self.data_module.train_dataloader())
        optimizer: Optimizer = instantiate(self.config.optimizer, params=params)
        scheduler = instantiate(
            self.config.scheduler, optimizer=optimizer, steps_per_epoch=steps_per_epoch
        )
        return [optimizer], [scheduler]

    def loss_fn(self, logits: Tensor, labels_ids: Tensor) -> Tensor:
        return F.cross_entropy(logits, labels_ids)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        words_ids, definitions_ids, attention_mask = batch
        logits = self.model.predict_words(
            definitions_ids, attention_mask=attention_mask
        )
        loss = self.loss_fn(logits, words_ids)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        words_ids, definitions_ids, attention_mask = batch
        logits = self.model.predict_words(
            definitions_ids, attention_mask=attention_mask
        )
        loss = self.loss_fn(logits, words_ids)
        self.log("val_loss", loss)
        return loss

    # train your model
    def fit(self) -> None:
        self.trainer.fit(self, self.data_module)
        self.log_hyperparams()
        self.log_cwd()
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.log_artifact("main.log")

    @rank_zero_only
    def evaluate(self):
        prev_device = self.device
        self.to(self.trainer.accelerator_connector.root_gpu)
        self.eval()

        metrics = {}
        metrics["d2w"] = self.def2word_evaluator(self.model)
        metrics["sts"] = self.sts_evaluator(
            encoder=self.encode,
            do_whitening=self.config.sts.do_whitening,
            to_lower=self.config.sts.to_lower,
        )
        self.log_main_metrics(metrics)

        metrics_str = OmegaConf.to_yaml(OmegaConf.create(metrics))
        metrics_path = Path("./metrics.yaml")
        metrics_path.write_text(metrics_str)
        self.log_artifact(metrics_path)

        self.to(prev_device)

    # run your whole experiments
    def run(self):
        self.fit()
        self.evaluate()

    def log_artifact(self, artifact_path: str) -> None:
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)

    def log_hyperparams(self) -> None:
        self.logger.log_hyperparams(
            {
                "model_name": self.config.model_name,
                "pooling_name": self.config.pooling_name,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "optimizer": self.config.optimizer._target_,
                "lr_scheduler": self.config.scheduler._target_,
            }
        )

    def log_cwd(self) -> None:
        self.logger.log_hyperparams({"_cwd": str(Path.cwd())})

    def log_main_metrics(self, metrics: Dict) -> None:
        main_metrics = {
            "d2w_test_MRR": metrics["d2w"]["test"]["MRR"],
            "d2w_test_top1": metrics["d2w"]["test"]["top1"],
            "d2w_test_top3": metrics["d2w"]["test"]["top3"],
            "d2w_test_top10": metrics["d2w"]["test"]["top10"],
            "sts_12": metrics["sts"]["STS12"]["spearman"]["cosine"],
            "sts_13": metrics["sts"]["STS13"]["spearman"]["cosine"],
            "sts_14": metrics["sts"]["STS14"]["spearman"]["cosine"],
            "sts_15": metrics["sts"]["STS15"]["spearman"]["cosine"],
            "sts_16": metrics["sts"]["STS16"]["spearman"]["cosine"],
            "sts_B": metrics["sts"]["STSB"]["spearman"]["cosine"],
            "sts_SICK-R": metrics["sts"]["SICK-R"]["spearman"]["cosine"],
            "sts_AVG": metrics["sts"]["AVG"]["spearman"]["cosine"],
        }
        self.logger.log_metrics(main_metrics)

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: Optional[int]) -> Tensor:
        inputs = self.tokenizer(
            sentences, padding=True, return_tensors="pt", truncation=True,
        )
        data_loader = torch.utils.data.DataLoader(
            list(zip(inputs.input_ids, inputs.attention_mask)),
            batch_size=batch_size or self.config.batch_size,
            num_workers=os.cpu_count(),
        )

        all_embs = []
        for batch in data_loader:
            sentence_ids, attention_mask = self.transfer_batch_to_device(
                batch, self.device
            )
            embs = self.model(sentence_ids, attention_mask=attention_mask).cpu()
            all_embs.append(embs)

        embeddings = torch.cat(all_embs, dim=0)
        return embeddings

    def save_model(self) -> None:
        self.model.pretrained_model.save_pretrained("./pretrained")
        self.tokenizer.save_pretrained("./pretrained")
