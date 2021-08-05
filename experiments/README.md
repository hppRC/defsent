# DefSent: Sentence Embeddings using Definition Sentences / Experiments

arxiv: [https://arxiv.org/abs/2105.04339](https://arxiv.org/abs/2105.04339)

## Installation

```bash
poetry install
```

## Download datasets and run pre-process

```bash
bash ./scripts/download-dataset.sh
poetry run python src/scripts/extract_data_from_ishiwatari.py
```


## Run an experiment

```bash
poetry run python main.py save_model=True model_name=bert-base-uncased pooling_name=CLS
```

For more detailed configurations, see `configs` directory.
We use [hydra](https://github.com/facebookresearch/hydra) for configurations.


## Start Mlflow Server

```bash
poetry run mlflow ui
# access http://127.0.0.1:5000
```


## Run Formatter

```bash
poetry run pysen run format
```

## Share models

```
<!-- example -->
huggingface-cli repo create defsent-bert-base-uncased-cls
git clone https://huggingface.co/cl-nagoya/defsent-bert-base-uncased-cls
mv /path/to/saved_model/* ./defsent-bert-base-uncased-cls/
cd ./defsent-bert-base-uncased-cls/
git add -A
git commit -m ":tada: Add pre-trained model"
```