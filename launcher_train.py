import argparse

from src.training import train_eogar as experiment
from evaluation.utils import generate_base_command, generate_run_commands



NAME = "Training EOgar"
MODE = "local"      # "local", "euler"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 1024

#
#   Model parameters and corresponding sizes
#

applicable_configs = {
    "seed": [i for i in range(1)],
    "digits": [3],
    "train_sizes": [64],
    "test_sizes": [16],
    "batch_size": [1],      # Has to be 1 for now because of the sequence length problem!
    "model_spec": [
        # Model size: small
        { "model_dimension": 64, "num_heads_encoder": 4, "n_encoder_blocks": 16 },

        # Model size: 7 Million
        # { "model_dimension": 64, "num_heads_encoder": 4, "num_heads_decoder": 4, "n_encoder_blocks": 64, "n_decoder_blocks": 64 },

        # Model size: 15 Million
        # { "model_dimension": 64, "num_heads_encoder": 4, "num_heads_decoder": 4, "n_encoder_blocks": 128, "n_decoder_blocks": 128 },
        # Model size: 30 Million
        # { "model_dimension": 64, "num_heads_encoder": 4, "num_heads_decoder": 4, "n_encoder_blocks": 256, "n_decoder_blocks": 256 },
        # Model size: 60 Million
        # { "model_dimension": 64, "num_heads_encoder": 4, "num_heads_decoder": 4, "n_encoder_blocks": 512, "n_decoder_blocks": 512 },

        # Model size: 7 Million
        # { "model_dimension": 64, "num_heads_encoder": 8, "num_heads_decoder": 8, "n_encoder_blocks": 64, "n_decoder_blocks": 64 },
        # Model size: 7 Million
        #{ "model_dimension": 64, "num_heads_encoder": 16, "num_heads_decoder": 16, "n_encoder_blocks": 64, "n_decoder_blocks": 64 },

        # Model size: 30 Million
        # { "model_dimension": 128, "num_heads_encoder": 4, "num_heads_decoder": 4, "n_encoder_blocks": 64, "n_decoder_blocks": 64 },
    ],
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
                        for learning_rate in applicable_configs["learning_rate"]:
                            for epochs in applicable_configs["epochs"]:
                                for seed in applicable_configs["seed"]:
                                    flags = {
                                        "name": NAME,
                                        "model_name": "EOgar",
                                        "digits": digits,
                                        "train_size": train_size,
                                        "test_size": test_size,
                                        "eval_size": EVAL_SIZE,
                                        "batch_size": batch_size,
                                        "model_dimension": model_spec["model_dimension"],
                                        "num_heads_encoder": model_spec["num_heads_encoder"],
                                        "n_encoder_blocks": model_spec["n_encoder_blocks"],
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
