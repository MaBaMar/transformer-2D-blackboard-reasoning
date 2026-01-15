# Transformer-2D-Blackboard-Reasoning
Research repository for the DL 2025 course project. The goal is to train transformers to use a 2D blackboard for better reasoning capabilities.

## Branches
`main:` contains current state of repo
`correction_train_eval:` Contains the newest error correction implementation attempt (not merged with main because of breaking changes). Use this branch if you want to investigate our error correction experiments. 

## Setup

Clone the repository

```
git clone git@github.com:MaBaMar/transformer-2D-blackboard-reasoning.git
```

Navigate to the root directory and install the requirements

```
bash build_env.sh
```

## Launch experiments

The project is organised into launcher scripts which will automatically deploy runs locally or on the cluster.

```
usage: launcher_train_eogar.py [-h] [--num-cpus NUM_CPUS] [--num-gpus NUM_GPUS] [--num-hours NUM_HOURS]

options:
  -h, --help            show this help message and exit
  --num-cpus NUM_CPUS
  --num-gpus NUM_GPUS
  --num-hours NUM_HOURS
```

The default values are one CPU, one GPU and a runtime of `7:59:00`.

Make sure to set the correct `MODE`at the top of the launcher before running it. Use `"dinfk"` to run it on the student cluster or `"local"` to run it locally.

### Train Models

To launch the training of EOgar or the CoT baseline execute the corresponding launcher script

```
python -m launchers.launcher_train_eogar
```
```
python -m launchers.launcher_train_cot
```

The models will be saved in the `./models/` directory.

To launch the training for the location generalization experiment use

```
python -m launchers.launcher_train_location_generalization
```

### Run the Experiments

To launch the out of distribution experiments you can use 

```
python -m launchers.launcher_eval_out_of_dist
```

To launch the location generalization experiments you can use

```
python -m launchers.launcher_eval_location_generalization
```

### Training models directly
The scripts `train_eogar.py` and `train_gptbase.py` in src/training as well as `run_evaluations.py` in evaluation/ can also be run directly.
For `train_eogar.py`:
```
usage: python -m src.training.train_eogar [-h] [--name NAME] [--model_name MODEL_NAME] --digits DIGITS --train_size TRAIN_SIZE --test_size TEST_SIZE --eval_size EVAL_SIZE
                                          --batch_size BATCH_SIZE --bb_height BB_HEIGHT --bb_width BB_WIDTH [--bb_randomize_position] --operation OPERATION
                                          --model_dimension MODEL_DIMENSION --num_heads_encoder NUM_HEADS_ENCODER --n_encoder_blocks N_ENCODER_BLOCKS --rope_mode ROPE_MODE
                                          --learning_rate LEARNING_RATE [--entropy_coef ENTROPY_COEF] --epochs EPOCHS [--error_pool_fraction ERROR_POOL_FRACTION]
                                          [--errors_per_epoch ERRORS_PER_EPOCH] [--error_correction] [--seed SEED] [--logging LOGGING] [--use_lr_scheduler]
                                          [--warmup_steps WARMUP_STEPS] [--num_sched_cycles NUM_SCHED_CYCLES] [--model_save_path_suffix MODEL_SAVE_PATH_SUFFIX]
```

For `train_gptbase.py`:
```
usage: python -m src.training.train_gptbase [-h] --name NAME [--model_name MODEL_NAME] --dataset_variant {scratchpad,cot} --operation {mixed,add,sub}
                                            --train_size TRAIN_SIZE --test_size TEST_SIZE --eval_size EVAL_SIZE --digits DIGITS --batch_size BATCH_SIZE
                                            --max_context_length MAX_CONTEXT_LENGTH --max_output_length MAX_OUTPUT_LENGTH --model_dimension MODEL_DIMENSION
                                            --num_heads NUM_HEADS --n_decoder_blocks N_DECODER_BLOCKS [--link_weights] --learning_rate LEARNING_RATE --epochs EPOCHS
                                            [--seed SEED] [--logging LOGGING] [--use_lr_scheduler] [--warmup_steps WARMUP_STEPS] [--num_sched_cycles NUM_SCHED_CYCLES]
                                            [--model_save_path_suffix MODEL_SAVE_PATH_SUFFIX]
```

For `run_evaluations.py`:
```
usage: python -m evaluation.run_evaluations [-h] [--name NAME] [--model_name MODEL_NAME] [--model_path MODEL_PATH] [--task TASK] [--digits DIGITS] [--size SIZE]
                                            [--batch_size BATCH_SIZE] [--height HEIGHT] [--width WIDTH] [--randomize_position] --operation OPERATION [--seed SEED]
                                            [--logging LOGGING]
```

All of the above must be run from the project root.

## Structure

Each file has its own description in its file head. Here a cannonical overview.

### projectlib
Contains helper functions and serves as the project code library.
You can install it into your virtual environment using pip, but installation is deprecated an no longer required. Running the project as python module `-m` from the source directory is sufficient for the library to be integrated.
```
pip install -e ./projectlib
```
This will install the project library as an editable package.

The folder and file structure is as follows:
```
projectlib
├── __init__.py
├── my_datasets        # contains implementations for synthesizing our custom datasets
│   ├── additions.py   # base dataset used for testing with pretrained models
│   ├── base.py        # base interface for all custom datasets, IMPORTANTLY: contains sampling logic
│   ├── _blackboard_operands.py # extensible types for the blackboard dataset, allows 
                                # defining additional operations than addition and subtraction
│   ├── blackboards.py  # core logic of blackboard chain generation
│   ├── collators.py    # efficient collators for our custom data formats
│   ├── cot.py          # simplified chain-of-thought implementation of scratchpads to save compute
│   ├── __init__.py     # required for proper module import
│   ├── _operands.py    # types for additions.py and the scratchpad datasets
│   ├── scratchpads.py  # full scratchpad chain-of-thought dataset implementation as described in (Nye et al., 2021)
│   └── utils.py        # helper functions
├── pyproject.toml      # installer, deprecated
├── pyrightconfig.json  # pyrightconfig for typecheckers (not longer maintained)
├── trainutils.py       # utility functions for training, especially monitoring
├── transformer         # 
│   ├── __init__.py     # python modules
│   └── tpe2d_model.py  # contains the core logic of 2D-TPE for our models, closely inspired by  (Li et al.,2024)
├── utils.py            # common utilities
└── wrappertypes.py     # interfaces for wrappers and result extraction
```

### src
Contains the implementations of the result extractor as well as all models (based on `projectlib/transformer/tpe2d_model.py`)

```
src
├── evaluation                  # contains wrappers that simulate real-world model deployment, including result extraction
│   ├── bb_chain_wrapper.py     # wrapper and result extraction for blackboard (EOgar) models
│   ├── gptbase_wrapper.py      # wrapper and result extraction for GPT base model (CoT)
├── __init__.py
├── models                      # model implementations
│   ├── edgar.py                # old and deprecated 2D blackboard model implementation based on an encoder-decoder architecture
│   ├── eogar.py                # EOgar 1D and 2D implementations
│   ├── gptbase.py              # implementation of the GPT style baseline model
└── training
    ├── train_eogar.py          # trainer for eogar. Launchable from the project root (see above)
    └── train_gptbase.py        # trainer for the CoT baseline. Launchable from the project root (see above)
```

### testscripts
Contains some unit tests for our code, but is mainly intended for testing of the devs and visual inspection
```
testscripts
├── test_blackboard.py        # unit test for the blackboard dataset and trivial sampler guarantees
├── test_gptbase.py           # visual test for the GPT style baseline model
├── test_scratchpad.py        # unit test for scratchpad and CoT datasets
└── wrapper_playground.py     # visual test and playground for interaction with a deployed model
```

### evaluation
Contains an evaluation runner which employs the wrappers for accuracy testing on full computation chains
```
evaluation
├── __init__.py
├── run_evaluations.py      # the script. For launching see above
└── utils.py                # helpers
```

### plots
Over the course of the project, we have performed more than 2500 logged runs on weights and biases.
This directory contains scripts we used for fetching and summarizing run results for the paper as well as some raw results.
```
plots
├── data
│   ├── OoD_Evaluation.csv              # raw data of the final OoD evaluation experiment runs
│   └── randomization_expermiment.csv   # raw data of the final randomization experiment runs
├── ood_table.py                        # python script to generate the latex table for the OOD results.
├── randomization_experiment.py         # python script to generate the latex table for the randomization results.
└── tables
    ├── ood_evaluation_top3.txt         # top 3 seeds for OOD (used for analysis)
    ├── ood_evaluation_top4.txt         # top 4 seeds for OOD (used for analysis)
    ├── ood_evaluation_top5.txt         # top 5 seeds for OOD (used for analysis)
    ├── ood_evaluation.txt              # OOD result table (used for analysis)
    └── random_exp.txt                  # full randomization result table. The paper contains a trimmed version
```

### launchers
Used to deploy experiments locally or on the cluster, as outlined above. Launching is strictly serial.
```
launchers
├── launcher_eval_location_generalization.py    # evaluates trained models with and without randomization. Should be launches AFTER launcher_train_location_generalization
├── launcher_eval_out_of_dist.py                # evaluated models on OOD settings, assumes trained models are available
├── launcher_train_cot.py            # trains GPT sytle baseline models on a specifiable dataset
├── launcher_train_eogar.py          # trains EOgar models
└── launcher_train_location_generalization.py # batch trains models for the OOD experiment
```
