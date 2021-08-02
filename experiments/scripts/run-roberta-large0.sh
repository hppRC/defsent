poetry run python main.py -m save_model=True gpus=[2] experiment_name=RoBERTa-large-CLS model_name=roberta-large pooling_name=CLS lr=4e-06 +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[2] experiment_name=RoBERTa-large-Mean model_name=roberta-large pooling_name=Mean lr=4e-06 +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[2] experiment_name=RoBERTa-large-Max model_name=roberta-large pooling_name=Max lr=5.656854249492381e-06 +exp_times=0,1,2,3,4