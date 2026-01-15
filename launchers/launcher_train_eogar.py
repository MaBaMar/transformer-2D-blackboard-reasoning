# ------------------------------------------------------------
# launcher_train_eogar.py
#
# Configurable runner for training EOgar type models. Extensively
# used for the OOD experiment described in the paper
#
# See README for directions on usage
# ------------------------------------------------------------


import argparse

from src.training import train_eogar as experiment
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "OoD_Train_EOgar_Debug"
MODE = "dinfk"      # "local", "euler", "dinfk"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 8192
NUM_SEEDS = 10

#
#   Model parameters and corresponding sizes
#

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],
    "digits": [10],
    "train_sizes": [8192],
    "test_sizes": [1024],
    "batch_size": [64],
    "model_spec": [
        { "model_name": "EOgar", "model_dimension": 64, "num_heads_encoder": 4, "n_encoder_blocks": 4 },
    ],
    "bb_specs": [
        { "height": 6, "width": 20, "randomize_position": True, "operation": "add" },
        { "height": 6, "width": 20, "randomize_position": True, "operation": "sub" },
        { "height": 6, "width": 20, "randomize_position": True, "operation": "mixed" },
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
                                                flags = {
                                                    "name": NAME,
                                                    "model_name": model_spec["model_name"] + f"-{rope_mode}",
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

                                                    "model_save_path_suffix": f"_op{bb_spec['operation']}",
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
