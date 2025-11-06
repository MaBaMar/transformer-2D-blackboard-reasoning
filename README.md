# Transformer-2D-Blackboard-Reasoning
Research repository for the DL 2025 course project. The goal is to train transformers to use a 2D blackboard for better reasoning capabilities.

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
Contains scripts to generate datasets and the generated datasets (preferrably as pytorch datasets)
