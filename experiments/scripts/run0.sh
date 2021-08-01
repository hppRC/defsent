poetry run python main.py -m save_model=True gpus=[2] experiment_name=BERT-base-CLS model_name=bert-base-uncased pooling_name=CLS +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[2] experiment_name=BERT-base-Mean model_name=bert-base-uncased pooling_name=Mean +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[2] experiment_name=BERT-base-Max model_name=bert-base-uncased pooling_name=Max +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[2] experiment_name=RoBERTa-base-CLS model_name=roberta-base pooling_name=CLS +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[2] experiment_name=RoBERTa-base-Mean model_name=roberta-base pooling_name=Mean +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[2] experiment_name=RoBERTa-base-Max model_name=roberta-base pooling_name=Max +exp_times=0,1,2,3,4