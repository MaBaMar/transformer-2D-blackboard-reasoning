import argparse

from src.training import train_eogar_old as experiment
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "Modelsize_EOgar"
MODE = "euler"      # "local", "euler"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 8192
NUM_SEEDS = 10

#
#   Model parameters and corresponding sizes
#

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],
    "digits": [4, 8, 12],
    "train_sizes": [4096],
    "test_sizes": [1024],
    "batch_size": [64],
    "model_spec": [
        # Vary encoder blocks
        { "model_name": "EOgar-d32-h4-b4", "model_dimension": 32, "num_heads_encoder": 4, "n_encoder_blocks": 4 },
        { "model_name": "EOgar-d32-h4-b8", "model_dimension": 32, "num_heads_encoder": 4, "n_encoder_blocks": 8 },
        { "model_name": "EOgar-d32-h4-b16", "model_dimension": 32, "num_heads_encoder": 4, "n_encoder_blocks": 16 },
        { "model_name": "EOgar-d32-h4-b32", "model_dimension": 32, "num_heads_encoder": 4, "n_encoder_blocks": 32 },
        
        { "model_name": "EOgar-d64-h4-b4", "model_dimension": 64, "num_heads_encoder": 4, "n_encoder_blocks": 4 },
        { "model_name": "EOgar-d64-h4-b8", "model_dimension": 64, "num_heads_encoder": 4, "n_encoder_blocks": 8 },
        { "model_name": "EOgar-d64-h4-b16", "model_dimension": 64, "num_heads_encoder": 4, "n_encoder_blocks": 16 },

        # Vary number of heads
        { "model_name": "EOgar-d32-h4-b4", "model_dimension": 32, "num_heads_encoder": 4, "n_encoder_blocks": 4 },
        { "model_name": "EOgar-d32-h16-b4", "model_dimension": 32, "num_heads_encoder": 16, "n_encoder_blocks": 4 },
        { "model_name": "EOgar-d32-h32-b4", "model_dimension": 32, "num_heads_encoder": 32, "n_encoder_blocks": 4 },
        { "model_name": "EOgar-d32-h64-b4", "model_dimension": 32, "num_heads_encoder": 64, "n_encoder_blocks": 4 },
        
        { "model_name": "EOgar-d64-h4-b4", "model_dimension": 64, "num_heads_encoder": 4, "n_encoder_blocks": 4 },
        { "model_name": "EOgar-d64-h16-b4", "model_dimension": 64, "num_heads_encoder": 16, "n_encoder_blocks": 4 },
        { "model_name": "EOgar-d64-h32-b4", "model_dimension": 64, "num_heads_encoder": 32, "n_encoder_blocks": 4 },

        # Vary both
        { "model_name": "EOgar-d32-h8-b8", "model_dimension": 32, "num_heads_encoder": 8, "n_encoder_blocks": 8 },
        { "model_name": "EOgar-d32-h16-b16", "model_dimension": 32, "num_heads_encoder": 16, "n_encoder_blocks": 16 },

        { "model_name": "EOgar-d64-h8-b8", "model_dimension": 64, "num_heads_encoder": 8, "n_encoder_blocks": 8 },
        { "model_name": "EOgar-d64-h16-b16", "model_dimension": 64, "num_heads_encoder": 16, "n_encoder_blocks": 16 },
    ],
    "bb_specs": [
        { "height": 8, "width": 16, "randomize_position": "false", "operation": "addition" },
        # { "height": 8, "width": 16, "randomize_position": "true", "operation": "addition" },
    ],
    "rope_mode": ["2d"],
    "learning_rate": [1e-3],
    "epochs": [8],
}

def main(args):
    command_list = []
    for digits in applicable_configs["digits"]:
        for train_size in applicable_configs["train_sizes"]:
            for test_size in applicable_configs["test_sizes"]:
                for batch_size in applicable_configs["batch_size"]:
                    for model_spec in applicable_configs["model_spec"]:
                        for bb_spec in applicable_configs["bb_specs"]:
                            for rope_mode in applicable_configs["rope_mode"]:
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
                                                "bb_height": bb_spec["height"],
                                                "bb_width": bb_spec["width"],
                                                "bb_randomize_position": bb_spec["randomize_position"],
                                                "operation": bb_spec["operation"],
                                                "model_dimension": model_spec["model_dimension"],
                                                "num_heads_encoder": model_spec["num_heads_encoder"],
                                                "n_encoder_blocks": model_spec["n_encoder_blocks"],
                                                "rope_mode": rope_mode,
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
