import argparse

from evaluation import run_evaluations as experiment
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "Evaluation_test"
MODE = "euler"      # "local", "euler"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 1024    # Make sure that this matches with the one in the training launchers
NUM_SEEDS = 1

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],
    "models": [
        # Base
        # { "name": "todo", "path": "todo", "task": "basic" },

        # Scratchpad
        # { "name": "todo", "path": "todo", "task": "scratchpad" },

        # 1D-RoPE
        # { "name": "todo", "path": "todo", "task": "blackboard-1d" },

        # 2D-RoPE
        { "name": "EOgar-100K", "path": "models/EOgar-100K-2d_e1_s64_d3.pt", "task": "blackboard-2d" },
    ],
    "digits": [3, 5],
}

def main(args):
    command_list = []
    for model in applicable_configs["models"]:
        for digits in applicable_configs["digits"]:
            for seed in applicable_configs["seed"]:
                flags = {
                    "name": NAME,
                    "model_name": model["name"],
                    "model_path": model["path"],
                    "task": model["task"],
                    "digits": digits,
                    "size": EVAL_SIZE,
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
