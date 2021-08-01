poetry run python main.py -m save_model=True gpus=[1] experiment_name=BERT-large-CLS model_name=bert-large-uncased pooling_name=CLS +exp_times=5,6,7,8,9
poetry run python main.py -m save_model=True gpus=[1] experiment_name=BERT-large-Mean model_name=bert-large-uncased pooling_name=Mean +exp_times=5,6,7,8,9
poetry run python main.py -m save_model=True gpus=[1] experiment_name=BERT-large-Max model_name=bert-large-uncased pooling_name=Max +exp_times=5,6,7,8,9
poetry run python main.py -m save_model=True gpus=[1] experiment_name=RoBERTa-large-CLS model_name=roberta-large pooling_name=CLS +exp_times=5,6,7,8,9
poetry run python main.py -m save_model=True gpus=[1] experiment_name=RoBERTa-large-Mean model_name=roberta-large pooling_name=Mean +exp_times=5,6,7,8,9
poetry run python main.py -m save_model=True gpus=[1] experiment_name=RoBERTa-large-Max model_name=roberta-large pooling_name=Max +exp_times=5,6,7,8,9