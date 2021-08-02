poetry run python main.py -m save_model=True gpus=[0] experiment_name=BERT-large-CLS model_name=bert-large-uncased pooling_name=CLS lr=5.656854249492381e-06 +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[0] experiment_name=BERT-large-Mean model_name=bert-large-uncased pooling_name=Mean lr=1.1313708498984761e-05 +exp_times=0,1,2,3,4
poetry run python main.py -m save_model=True gpus=[0] experiment_name=BERT-large-Max model_name=bert-large-uncased pooling_name=Max lr=8e-06 +exp_times=0,1,2,3,4
