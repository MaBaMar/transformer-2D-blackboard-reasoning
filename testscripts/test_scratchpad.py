# ------------------------------------------------------------
# test_scratchpad.py
#
# Test for scratchpads.py and associated modules.
#
# Run with:
#  - python -m testscripts.test_scratchpads
#  - run from the project root
# ------------------------------------------------------------

from projectlib.my_datasets.scratchpads import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments



TESTS = [
    (29, 57, "+", "Input: 2 9 + 5 7 <sep>\nComputation:\n2 9 + 5 7 , C: 0\n2 + 5 , 6 C: 1 # added 9 + 7 = 6 carry 1\n, 8 6 C: 0 # added 2 + 5 + 1 = 8 carry 0\n0 8 6\n\nResult: 8 6\n<eos>"),
    (83, 76, "+", "Input: 8 3 + 7 6 <sep>\nComputation:\n8 3 + 7 6 , C: 0\n8 + 7 , 9 C: 0 # added 3 + 6 = 9 carry 0\n, 5 9 C: 1 # added 8 + 7 = 5 carry 1\n1 5 9\n\nResult: 1 5 9\n<eos>"),
    (123, 169, "+", "Input: 1 2 3 + 1 6 9 <sep>\nComputation:\n1 2 3 + 1 6 9 , C: 0\n1 2 + 1 6 , 2 C: 1 # added 3 + 9 = 2 carry 1\n1 + 1 , 9 2 C: 0 # added 2 + 6 + 1 = 9 carry 0\n, 2 9 2 C: 0 # added 1 + 1 = 2 carry 0\n0 2 9 2\n\nResult: 2 9 2\n<eos>"),
    (62, 49, "-", "Input: 6 2 - 4 9 <sep>\nComputation:\n6 2 - 4 9 , C: 0\n6 - 4 , 3 C: 1 # subtracted 9 + 3 = 12 carry 1\n, 1 3 C: 0 # subtracted 4 + 1 + 1 = 6 carry 0\n1 3\n\nResult: 1 3\n<eos>"),
    (69, 42, "-", "Input: 6 9 - 4 2 <sep>\nComputation:\n6 9 - 4 2 , C: 0\n6 - 4 , 7 C: 0 # subtracted 2 + 7 = 9 carry 0\n, 2 7 C: 0 # subtracted 4 + 2 = 6 carry 0\n2 7\n\nResult: 2 7\n<eos>"),
]



def run_test(a, b, op, res):
    dataset = ScratchpadDataset(
        path="datasets/test_scratchpad.pt",
        generation_spec=GenerationSpec(1000, 10000, 1),
        operand=op,
    )

    scratchpad = dataset._generate_scratchpad(a, b)

    if scratchpad == res:
        return False
    
    print(30 * "-")

    print("\033[91mTest failed!\033[0m")
    print(f"\na={a}, b={b} op={op}\n\n")
    
    print("Dataset:\n")
    print(scratchpad)
    print("\nReference:\n")
    print(res)

    print(30 * "-")

    return True



if __name__ == "__main__":
    failed = False
    for a, b, op, res in TESTS:
        failed = failed or (True if run_test(a, b, op, res) else False)

    if not failed:
        print("\033[92mAll tests passed!\033[0m")

