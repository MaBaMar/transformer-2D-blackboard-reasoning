# ------------------------------------------------------------
# launcher_eval_location_generalization.py
#
# Configurable runner for evaluating trained EOgar style
# models on position randomized and non-position randomized data.
# Used for randomization evaluation in our paper.
#
# See README for directions on usage
# ------------------------------------------------------------

import argparse

from evaluation import run_evaluations as experiment
from evaluation.utils import generate_base_command, generate_run_commands


NAME = "Location_Generalization_Eval"
MODE = "dinfk"      # "local", "euler", "dinfk"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 8192
BATCH_SIZE = 64
NUM_SEEDS = 10

# exp mixer generator
def mix(l1, l2, l3):
    for i in l1:
        for j in l2:
            for k in l3:
                yield (i, j, k)

bb_board_sizes = [(6, 20)]
operations = ["add", "sub"] # ["add", "sub"] # TODO: change to mixed
randomize_position = [True, False]

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],

    # Evaluate:
    # - CoT (single checkpoint per seed)
    # - EOgar trained on fixed positions (trainF -> _rF.pt)
    # - EOgar trained on randomized positions (trainT -> _rT.pt)
    "models": [
        # EOgar trained with TrainPos = false  -> _rF.pt
        # { "name": "EOgar-1d", "path": "models/EOgar-1d_d10_s{seed:d}_r{rand}_op{operation}_H{height}W{width}.pt", "task": "blackboard-1d" },
        { "name": "EOgar-2d", "path": "models/EOgar-2d_d10_s{seed:d}_r{rand}_op{operation}_H{height}W{width}.pt", "task": "blackboard-2d" },
    ],

    # Evaluate on BOTH fixed and randomized positions, for each operation
    "bb_specs": [
        *[{ "height": x[0], "width": x[1], "randomize_position": y, "operation": z } for x,y,z in mix(bb_board_sizes, randomize_position, operations)]
    ],

    "digits": [10, 11, 12],
}

def main(args):
    command_list = []

    for model in applicable_configs["models"]:
        for bb_spec in applicable_configs["bb_specs"]:
            for digits in applicable_configs["digits"]:
                for seed in applicable_configs["seed"]:
                    for train_pos in [True, False]:

                        run_name = (
                            f"{NAME}_{model['name']}"
                            f"_Op{bb_spec['operation']}"
                            f"_TrainPos{train_pos}"
                            f"_EvalPos{bb_spec['randomize_position']}"
                            f"_BDimH{bb_spec['height']}W{bb_spec['width']}"
                            f"_d{digits}_s{seed}"
                        )

                        model_path = model["path"].format(
                            seed=seed,
                            operation=bb_spec["operation"],
                            rand='T' if train_pos else 'F',
                            height=bb_spec["height"],
                            width=bb_spec["width"]
                        )

                        flags = {
                            "name": run_name,
                            "model_name": model["name"],
                            "model_path": model_path,
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
    parser.add_argument("--num-hours", type=int, default=4)
    args = parser.parse_args()
    main(args)
