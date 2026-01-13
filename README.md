# Transformer-2D-Blackboard-Reasoning
Research repository for the DL 2025 course project. The goal is to train transformers to use a 2D blackboard for better reasoning capabilities.

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

## Structure
Please update this section when extending the project folder.

### projectlib
Contains helper functions and serves as the project code library. Everything that is reused between models/runs should go there.
You can install it into your virtual environment using pip:
```
pip install -e ./projectlib
```
This will install the project library as an editable package, allowing you to make changes to the code and see them reflected in your project without having to reinstall the package each time. It also allows you to import the library in arbitrary location using e.g. `import projectlib`.

To run stuff, always use the module running approach. E.g. to run test_blackboard.py:
```
python -m projectlib.test_blackboard
```
This ensures all modules and libraries are properly found.

### datasets
Contains scripts to generate datasets and the generated datasets (preferrably as pytorch datasets).

Run this command to generate datasets with a specific amount of digits:

```
python -m projectlib.datagen --digits 5
```

This will generate evaluation datasets for the baselines and a training and evaluation dataset for the blackboards.
