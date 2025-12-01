import argparse

from src.training import train_edgar as experiment
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "Training Edgar"
MODE = "local"      # "local", "euler"
LOGGING = "wandb"   # "wandb", "local", "none"

applicable_configs = {
    "seed": [i for i in range(1)],
    "digits": [1],
    "train_sizes": [32],
    "eval_sizes": [8],
    "batch_size": [4],
    "model_dimension": [64],
    "learning_rate": [1e-3],
    "epochs": [8],
}

def main(args):
    command_list = []
    for digits in applicable_configs["digits"]:
        for train_size in applicable_configs["train_sizes"]:
            for eval_size in applicable_configs["eval_sizes"]:
                for batch_size in applicable_configs["batch_size"]:
                    for model_dimension in applicable_configs["model_dimension"]:
                        for learning_rate in applicable_configs["learning_rate"]:
                            for epochs in applicable_configs["epochs"]:
                                for seed in applicable_configs["seed"]:
                                    flags = {
                                        "name": NAME,
                                        "model_name": "Edgar",
                                        "digits": digits,
                                        "train_size": train_size,
                                        "eval_size": eval_size,
                                        "batch_size": batch_size,
                                        "model_dimension": model_dimension,
                                        "learning_rate": learning_rate,
                                        "epochs": epochs,
                                        "seed": seed,
                                        "logging": LOGGING,
                                    }
                                    cmd = generate_base_command(experiment, flags=flags)
                                    command_list.append(cmd)

    generate_run_commands(
        command_list,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        mode=MODE,
        num_hours=args.num_hours,
        promt=True,
        mem=16000,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-hours", type=int, default=8)
    args = parser.parse_args()
    main(args)
