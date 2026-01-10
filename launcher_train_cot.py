import argparse

from src.training import train_gptbase as experiment
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "OoD_Train_CoT"
MODE = "dinfk"      # "local", "euler", "dinfk"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 16384
NUM_SEEDS = 5

#
#   Model parameters and corresponding sizes
#

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],
    "digits": [10],
    "train_sizes": [16384],
    "test_sizes": [1024],
    "batch_size": [8],
    "model_spec": [
        # { "model_name": "CoT-d64-h4-b8", "model_dimension": 64, "num_heads": 4, "n_decoder_blocks": 8 },
        { "model_name": "CoT", "model_dimension": 64, "num_heads": 4, "n_decoder_blocks": 4 },
        # { "model_name": "CoT-d32-h4-b8", "model_dimension": 32, "num_heads": 4, "n_decoder_blocks": 8 },
    ],
    "operation": ["add", "sub", "mixed"],
    "dataset_variant": ["cot"], # ["cot", "scratchpad"],
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
                        for operation in applicable_configs["operation"]:
                            for dataset_variant in applicable_configs["dataset_variant"]:
                                for learning_rate in applicable_configs["learning_rate"]:
                                    for epochs in applicable_configs["epochs"]:
                                        for seed in applicable_configs["seed"]:
                                            flags = {
                                                "name": NAME,
                                                "model_name": model_spec["model_name"],
                                                "digits": digits,
                                                "train_size": train_size,
                                                "test_size": test_size,
                                                "eval_size": EVAL_SIZE,
                                                "batch_size": batch_size,
                                                "dataset_variant": dataset_variant,
                                                "operation": operation,
                                                "max_context_length": 400,
                                                "max_output_length": 400,
                                                "model_dimension": model_spec["model_dimension"],
                                                "num_heads": model_spec["num_heads"],
                                                "n_decoder_blocks": model_spec["n_decoder_blocks"],
                                                "learning_rate": learning_rate,
                                                "epochs": epochs,
                                                "seed": seed,
                                                "logging": LOGGING,

                                                "model_save_path_suffix": f"_op{operation}",
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
