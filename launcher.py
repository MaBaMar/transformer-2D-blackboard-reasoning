import argparse

from evaluation.experiment import *
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "Baselines"
MODE = "local"      # "local", "euler"
LOGGING = "wandb"   # "wandb", "local", "none"

applicable_configs = {
    "seed": [i for i in range(1)],
    "models": [
        "Llama-13B",     
    ],
    "task": [
        "addition",
        "scratchpad",
        #"blackboard",
    ],
    "digits": [3]
}

def main(args):
    command_list = []
    for model in applicable_configs["models"]:
        for task in applicable_configs["task"]:
            for digits in applicable_configs["digits"]:
                for seed in applicable_configs["seed"]:
                    flags = {
                        "name": NAME,
                        "model_name": model,
                        "task": task,
                        "digits": digits,
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