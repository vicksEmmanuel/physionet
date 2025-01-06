
import argparse
from physionet_paligemma import PhysiotherapyPaligemmaConfig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the Physiotherapy model")
    args = parser.parse_args()
    config_path = "config/physionet.yaml"
    physiotherapy = PhysiotherapyPaligemmaConfig(config_path)

    physiotherapy.eval(config_path=config_path)