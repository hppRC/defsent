from pathlib import Path

import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [" ".join(sent) if sent != [] else "." for sent in batch]
    embeddings = params["encoder"](batch)
    return embeddings


class SentEvalEvaluator:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __call__(self, encoder):
        # Set params for SentEval
        params_senteval = {"task_path": self.data_dir, "usepytorch": True, "kfold": 10}
        # params_senteval = {"task_path": self.data_dir, "usepytorch": True, "kfold": 2}
        params_senteval["classifier"] = {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            # "epoch_size": 1,
            "epoch_size": 4,
        }
        params_senteval["encoder"] = encoder

        se = senteval.engine.SE(params_senteval, batcher, prepare)

        # sts = [
        #     "STS12",
        #     "STS13",
        #     "STS14",
        #     "STS15",
        #     "STS16",
        #     "STSBenchmark",
        #     "SICKRelatedness",
        # ]
        classification_tasks = [
            "MR",
            "CR",
            "SUBJ",
            "MPQA",
            "SST2",
            "TREC",
            "MRPC",
            # "SICKEntailment",
        ]
        # probing_tasks = [
        #     "Length",
        #     "WordContent",
        #     "Depth",
        #     "TopConstituents",
        #     "BigramShift",
        #     "Tense",
        #     "SubjNumber",
        #     "ObjNumber",
        #     "OddManOut",
        #     "CoordinationInversion",
        # ]

        metrics = {}
        # for task in classification_tasks + probing_tasks + sts:
        for task in classification_tasks:
            # for task in se.list_tasks:
            print(task)
            try:
                metrics[task] = {
                    k: self.convert(v) for k, v in se.eval([task])[task].items()
                }
            except:
                print("error:", task)

        return metrics

    def convert(self, v):
        try:
            return float(v)
        except:
            try:
                return [float(x) for x in v]
            except:
                return -1
