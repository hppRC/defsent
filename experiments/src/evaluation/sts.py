from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from torch import Tensor
from tqdm import tqdm


# https://arxiv.org/pdf/2104.01767.pdf
def whitening_torch_final(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    # cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu) / embeddings.size(0)
    u, s, _ = torch.svd(cov)
    W = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings


class EmbeddingSimilarityEvaluator:
    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
        batch_size: int = 1024,
        name: str = "",
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores

        # print(name, len(self.sentences1))
        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.name = name
        self.batch_size = batch_size

    def __call__(
        self,
        encoder: Callable[[List[str]], Tensor],
        do_whitening: bool = False,
        to_lower: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        if to_lower:
            self.sentences1 = [x.lower() for x in self.sentences1]
            self.sentences2 = [x.lower() for x in self.sentences2]

        embeddings1 = encoder(self.sentences1, batch_size=self.batch_size)
        embeddings2 = encoder(self.sentences2, batch_size=self.batch_size)

        if do_whitening:
            num_pairs = embeddings1.shape[0]
            embeddings = whitening_torch_final(
                torch.cat([embeddings1, embeddings2], dim=0)
            )
            embeddings1 = embeddings[:num_pairs, :]
            embeddings2 = embeddings[num_pairs:, :]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [
            np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)
        ]

        # convert to a premitive float type
        eval_pearson = lambda my_score: float(pearsonr(self.scores, my_score)[0]) * 100
        eval_spearman = (
            lambda my_score: float(spearmanr(self.scores, my_score)[0]) * 100
        )

        return {
            "spearman": {
                "cosine": eval_spearman(cosine_scores),
                "manhattan": eval_spearman(manhattan_distances),
                "euclidean": eval_spearman(euclidean_distances),
                "dot": eval_spearman(dot_products),
            },
            "pearson": {
                "cosine": eval_pearson(cosine_scores),
                "manhattan": eval_pearson(manhattan_distances),
                "euclidean": eval_pearson(euclidean_distances),
                "dot": eval_pearson(dot_products),
            },
        }


class SICKRelatednessEvaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []

        with (data_dir / "SICK" / "SICK_test_annotated.txt").open() as f:
            _ = next(f)
            for line in f:
                _, sentence1, sentence2, score, *_ = line.strip().split("\t")
                sentences1.append(sentence1)
                sentences2.append(sentence2)
                scores.append(float(score))

        super().__init__(sentences1, sentences2, scores, name="sick-relatedness")


class STSBenchmarkEvaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, data_dir: Path):
        name = "sts-benchmark"

        datasets = [
            # "sts-train.csv",
            # "sts-dev.csv",
            "sts-test.csv",
        ]

        sentences1, sentences2, scores = [], [], []

        for dataset in datasets:
            with (data_dir / "stsbenchmark" / dataset).open() as f:
                for line in f:
                    _, _, _, _, score, sentence1, sentence2, *_ = line.strip().split(
                        "\t"
                    )
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(score))

        super().__init__(list(sentences1), list(sentences2), list(scores), name=name)


class STS2016Evaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, data_dir: Path):
        name = "sts-2016"

        sentences1, sentences2, scores = [], [], []
        datasets = [
            "answer-answer",
            "headlines",
            "plagiarism",
            "postediting",
            "question-question",
        ]

        for dataset in datasets:
            with (
                data_dir / "2016" / "test" / f"STS2016.gs.{dataset}.txt"
            ).open() as gs, (
                data_dir / "2016" / "test" / f"STS2016.input.{dataset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores, name=name)


class STS2015Evaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, data_dir: Path):
        name = "sts-2015"

        sentences1, sentences2, scores = [], [], []
        datasets = [
            "answers-forums",
            "answers-students",
            "belief",
            "headlines",
            "images",
        ]

        for dataset in datasets:
            with (data_dir / "2015" / "test" / f"STS.gs.{dataset}.txt").open() as gs, (
                data_dir / "2015" / "test" / f"STS.input.{dataset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores, name=name)


class STS2014Evaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, data_dir: Path):
        name = "sts-2014"

        sentences1, sentences2, scores = [], [], []
        datasets = [
            "deft-forum",
            "deft-news",
            "headlines",
            "images",
            "OnWN",
            "tweet-news",
        ]

        for dataset in datasets:
            with (data_dir / "2014" / "test" / f"STS.gs.{dataset}.txt").open() as gs, (
                data_dir / "2014" / "test" / f"STS.input.{dataset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores, name=name)


class STS2013Evaluator(EmbeddingSimilarityEvaluator):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, data_dir: Path):
        name = "sts-2013"

        sentences1, sentences2, scores = [], [], []
        datasets = ["FNWN", "headlines", "OnWN"]

        for dataset in datasets:
            with (data_dir / "2013" / "test" / f"STS.gs.{dataset}.txt").open() as gs, (
                data_dir / "2013" / "test" / f"STS.input.{dataset}.txt"
            ).open() as f:
                for line_input, line_gs, *_ in zip(f, gs):
                    sentence1, sentence2 = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores, name=name)


class STS2012Evaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, data_dir: Path):
        name = "sts-2012"

        sentences1, sentences2, scores = [], [], []
        datasets = [
            "MSRpar",
            "MSRvid",
            "SMTeuroparl",
            "surprise.OnWN",
            "surprise.SMTnews",
        ]

        for dataset in datasets:
            with (data_dir / "2012" / "test" / f"STS.gs.{dataset}.txt").open() as gs, (
                data_dir / "2012" / "test" / f"STS.input.{dataset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores, name=name)


class STSEvaluation:
    def __init__(self, data_dir: Union[str, Path]):
        data_dir = Path(data_dir)
        self.sts_evaluators = {
            "STS12": STS2012Evaluator(data_dir=data_dir),
            "STS13": STS2013Evaluator(data_dir=data_dir),
            "STS14": STS2014Evaluator(data_dir=data_dir),
            "STS15": STS2015Evaluator(data_dir=data_dir),
            "STS16": STS2016Evaluator(data_dir=data_dir),
            "STSB": STSBenchmarkEvaluator(data_dir=data_dir),
            "SICK-R": SICKRelatednessEvaluator(data_dir=data_dir),
        }

        self.metrics = ["spearman", "pearson"]
        self.methods = ["cosine", "manhattan", "euclidean", "dot"]

    @torch.no_grad()
    def __call__(
        self,
        encoder: Callable[[List[str]], Tensor],
        do_whitening: bool = False,
        to_lower: bool = False,
    ):
        sts_evaluations = {}
        for name, evaluator in tqdm(list(self.sts_evaluators.items())):
            sts_evaluations[name] = evaluator(
                encoder, do_whitening=do_whitening, to_lower=to_lower
            )

        sts_evaluations["AVG"] = {}
        for metric in self.metrics:
            sts_evaluations["AVG"][metric] = {}

            for method in self.methods:
                sts_evaluations["AVG"][metric][method] = 0.0

                for task in self.sts_evaluators:
                    sts_evaluations["AVG"][metric][method] += sts_evaluations[task][
                        metric
                    ][method]
                sts_evaluations["AVG"][metric][method] /= len(self.sts_evaluators)

        return sts_evaluations
