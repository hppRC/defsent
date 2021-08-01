import hydra
from omegaconf import DictConfig
from src.experiment import Experiment


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    exp = Experiment(config)
    ret = exp.run()

    if config.save_model:
        exp.save_model()

    return ret


if __name__ == "__main__":
    main()
