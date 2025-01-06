
import argparse
from physionet_paligemma import PhysiotherapyPaligemmaConfig
from physionet import Physiotherapy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the Physiotherapy model")
    parser.add_argument("--train", type=str, default="True", help="Path to the config file")
    parser.add_argument("--use-paligemma", type=str, default="True", help="Use Paligemma model")
    parser.add_argument("--utilize-memory", type=str, default="True", help="Utilize memory for training")

    args = parser.parse_args()

    if args.use_paligemma == "True":
        physiotherapy = PhysiotherapyPaligemmaConfig(
            "config/physionet.yaml"
        )

        if args.train == "True":
            physiotherapy.train()
        else:
            print("Proceeding to inferencing")
            physiotherapy.inference(
                "config/physionet.yaml", 
                "/home/featurize/work/physionet/ai/dataset/data/videos/Physiotherapy Actions Detection_ds0_Postural Analysis Anterior.mp4",
                "output.mp4"
            )
    else:
        physiotherapy = Physiotherapy()
        if args.train == "True":
            use_memory = args.utilize_memory == "True"
            physiotherapy.train("config/base.yaml", utilize_memory=use_memory)
        else:
            print("Proceeding to inferencing")
            physiotherapy.inference("config/base.yaml", "/home/featurize/work/physionet/ai/dataset/data/videos/Physiotherapy Actions Detection_ds0_Clinical Examination - GALS Screen - Gait, Arms, Legs, Spine - (OLD VIDEO) - Dr Gill.mp4")