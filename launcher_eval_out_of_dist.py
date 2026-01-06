import argparse

from evaluation import run_evaluations as experiment
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "OoD_Evaluation"
MODE = "dinfk"      # "local", "euler", "dinfk"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 8192    # Make sure that this matches with the one in the training launchers
BATCH_SIZE = 64
NUM_SEEDS = 5

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],
    "models": [
        # CoT
        { "name": "CoT", "path": "models/CoT_d10_s{seed:d}.pt", "task": "cot" },

        # 1D-RoPE
        { "name": "EOgar-1d", "path": "models/EOgar-1d_d10_s{seed:d}_rT.pt", "task": "blackboard-1d" },

        # 2D-RoPE
        { "name": "EOgar-2d", "path": "models/EOgar-2d_d10_s{seed:d}_rT.pt", "task": "blackboard-2d" },
    ],
    "bb_specs": [
        { "height": 6, "width": 20, "randomize_position": "false", "operation": "add" },
    ],
    "digits": [10, 11, 12, 13],
}

def main(args):
    command_list = []
    for model in applicable_configs["models"]:
        for bb_spec in applicable_configs["bb_specs"]:
            for digits in applicable_configs["digits"]:
                for seed in applicable_configs["seed"]:
                    flags = {
                        "name": NAME,
                        "model_name": model["name"],
                        "model_path": model["path"].format(seed=seed),
                        "task": model["task"],
                        "digits": digits,
                        "size": EVAL_SIZE,
                        "batch_size": BATCH_SIZE,
                        "height": bb_spec["height"],
                        "width": bb_spec["width"],
                        "randomize_position": bb_spec["randomize_position"],
                        "operation": bb_spec["operation"],
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
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-hours", type=int, default=7)
    args = parser.parse_args()
    main(args)
