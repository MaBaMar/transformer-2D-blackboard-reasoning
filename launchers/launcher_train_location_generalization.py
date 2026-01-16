# ------------------------------------------------------------
# launcher_train_location_generalization.py
#
# Configurable runner for training models on the location generalization
# setting described in the paper.
#
# See README for directions on usage
# ------------------------------------------------------------


import argparse

from src.training import train_eogar as experiment
from evaluation.utils import generate_base_command, generate_run_commands


NAME = "Location_Generalization_Train"
MODE = "dinfk"      # "local", "euler", "dinfk"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 8192
NUM_SEEDS = 10

# exp mixer generator
def mix(l1, l2, l3):
    for i in l1:
        for j in l2:
            for k in l3:
                yield (i, j, k)

bb_board_sizes = [(6, 20)]
operations =  ["add", "sub", "mixed"]
randomize_position = [True, False]

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],
    "digits": [10],
    "train_sizes": [8192],
    "test_sizes": [1024],
    "batch_size": [64],

    "model_spec": [
        { "model_name": "EOgar", "model_dimension": 64, "num_heads_encoder": 4, "n_encoder_blocks": 4 },
    ],

    # Train BOTH fixed + randomized positions for each operation
    "bb_specs": [
        *[{ "height": bb_size[0], "width": bb_size[1], "randomize_position": r, "operation": op } for bb_size, r, op in mix(bb_board_sizes, randomize_position, operations)]
    ],

    "entropy_coeff": [0.5],
    "rope_mode": ["1d", "2d"],

    "learning_rate": [1e-3],
    "epochs": [10],
}

def main(args):
    command_list = []

    for digits in applicable_configs["digits"]:
        for train_size in applicable_configs["train_sizes"]:
            for test_size in applicable_configs["test_sizes"]:
                for batch_size in applicable_configs["batch_size"]:
                    for model_spec in applicable_configs["model_spec"]:
                        for bb_spec in applicable_configs["bb_specs"]:
                            for entropy_coeff in applicable_configs["entropy_coeff"]:
                                for rope_mode in applicable_configs["rope_mode"]:
                                    for learning_rate in applicable_configs["learning_rate"]:
                                        for epochs in applicable_configs["epochs"]:
                                            for seed in applicable_configs["seed"]:

                                                model_name = model_spec["model_name"] + f"-{rope_mode}"

                                                run_name = (
                                                    f"{NAME}_{model_name}"
                                                    f"_Op{bb_spec['operation']}"
                                                    f"_TrainPos{bb_spec['randomize_position']}"
                                                    f"_BDimH{bb_spec['height']}W{bb_spec['width']}"
                                                    f"_d{digits}_s{seed}"
                                                )

                                                flags = {
                                                    "name": run_name,
                                                    "model_name": model_name,

                                                    "digits": digits,
                                                    "train_size": train_size,
                                                    "test_size": test_size,
                                                    "eval_size": EVAL_SIZE,
                                                    "batch_size": batch_size,

                                                    "bb_height": bb_spec["height"],
                                                    "bb_width": bb_spec["width"],
                                                    "bb_randomize_position": bb_spec["randomize_position"],
                                                    "operation": bb_spec["operation"],

                                                    "model_dimension": model_spec["model_dimension"],
                                                    "num_heads_encoder": model_spec["num_heads_encoder"],
                                                    "n_encoder_blocks": model_spec["n_encoder_blocks"],

                                                    "entropy_coef": entropy_coeff,
                                                    "rope_mode": rope_mode,

                                                    "learning_rate": learning_rate,
                                                    "epochs": epochs,

                                                    "seed": seed,
                                                    "logging": LOGGING,

                                                    "model_save_path_suffix": f"_op{bb_spec['operation']}_H{bb_spec['height']}W{bb_spec['width']}"
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
