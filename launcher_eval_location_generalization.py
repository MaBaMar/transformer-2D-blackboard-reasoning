import argparse

from evaluation import run_evaluations as experiment
from evaluation.utils import generate_base_command, generate_run_commands


NAME = "Location_Generalization_Eval"
MODE = "dinfk"      # "local", "euler", "dinfk"
LOGGING = "wandb"   # "wandb", "local", "none"

EVAL_SIZE = 8192
BATCH_SIZE = 64
NUM_SEEDS = 5

applicable_configs = {
    "seed": [i for i in range(NUM_SEEDS)],

    # Evaluate:
    # - CoT (single checkpoint per seed)
    # - EOgar trained on fixed positions (trainF -> _rF.pt)
    # - EOgar trained on randomized positions (trainT -> _rT.pt)
    "models": [
        { "name": "CoT", "path": "models/{operation}/CoT_d10_s{seed:d}.pt", "task": "cot" },

        # EOgar trained with TrainPos = false  -> _rF.pt
        { "name": "EOgar-1d-trainF", "path": "models/{operation}/EOgar-1d_d10_s{seed:d}_rF.pt", "task": "blackboard-1d" },
        { "name": "EOgar-2d-trainF", "path": "models/{operation}/EOgar-2d_d10_s{seed:d}_rF.pt", "task": "blackboard-2d" },

        # EOgar trained with TrainPos = true -> _rT.pt
        { "name": "EOgar-1d-trainT", "path": "models/{operation}/EOgar-1d_d10_s{seed:d}_rT.pt", "task": "blackboard-1d" },
        { "name": "EOgar-2d-trainT", "path": "models/{operation}/EOgar-2d_d10_s{seed:d}_rT.pt", "task": "blackboard-2d" },
    ],

    # Evaluate on BOTH fixed and randomized positions, for each operation
    "bb_specs": [
        # add
        { "height": 8, "width": 20, "randomize_position": "false", "operation": "add" },
        { "height": 8, "width": 20, "randomize_position": "true",  "operation": "add" },

        # sub
        { "height": 8, "width": 20, "randomize_position": "false", "operation": "sub" },
        { "height": 8, "width": 20, "randomize_position": "true",  "operation": "sub" },

        # mixed
        { "height": 8, "width": 20, "randomize_position": "false", "operation": "mixed" },
        { "height": 8, "width": 20, "randomize_position": "true",  "operation": "mixed" },
    ],

    "digits": [10],
}

def main(args):
    command_list = []

    for model in applicable_configs["models"]:
        for bb_spec in applicable_configs["bb_specs"]:
            for digits in applicable_configs["digits"]:
                for seed in applicable_configs["seed"]:

                    run_name = (
                        f"{NAME}_{model['name']}"
                        f"_Op{bb_spec['operation']}"
                        f"_EvalPos{bb_spec['randomize_position']}"
                        f"_d{digits}_s{seed}"
                    )

                    model_path = model["path"].format(
                        seed=seed,
                        operation=bb_spec["operation"],
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
