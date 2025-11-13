"""Script to test parts of the scratchpad implementation"""

from projectlib.my_datasets.scratchpads import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments



TESTS = [
    (29, 57, "+", "Input: 2 9 + 5 7\nTarget:\n<scratch>\n2 9 + 5 7 , C: 0\n2 + 5 , 6 C: 1 # added 9 + 7 = 6 carry 1\n, 8 6 C: 0 # added 2 + 5 + 1 = 8 carry 0\n0 8 6\n</scratch>\nResult: 8 6\n"),
    (83, 76, "+", "Input: 8 3 + 7 6\nTarget:\n<scratch>\n8 3 + 7 6 , C: 0\n8 + 7 , 9 C: 0 # added 3 + 6 = 9 carry 0\n, 5 9 C: 1 # added 8 + 7 = 5 carry 1\n1 5 9\n</scratch>\nResult: 1 5 9\n"),
    (123, 169, "+", "Input: 1 2 3 + 1 6 9\nTarget:\n<scratch>\n1 2 3 + 1 6 9 , C: 0\n1 2 + 1 6 , 2 C: 1 # added 3 + 9 = 2 carry 1\n1 + 1 , 9 2 C: 0 # added 2 + 6 + 1 = 9 carry 0\n, 2 9 2 C: 0 # added 1 + 1 = 2 carry 0\n0 2 9 2\n</scratch>\nResult: 2 9 2\n"),
    (62, 49, "-", "Input: 6 2 - 4 9\nTarget:\n<scratch>\n6 2 - 4 9 , C: 0\n6 - 4 , 3 C: 1 # subtracted 9 + 3 = 12 carry 1\n, 1 3 C: 0 # subtracted 4 + 1 + 1 = 6 carry 0\n1 3\n</scratch>\nResult: 1 3\n"),
    (69, 42, "-", "Input: 6 9 - 4 2\nTarget:\n<scratch>\n6 9 - 4 2 , C: 0\n6 - 4 , 7 C: 0 # subtracted 2 + 7 = 9 carry 0\n, 2 7 C: 0 # subtracted 4 + 2 = 6 carry 0\n2 7\n</scratch>\nResult: 2 7\n"),
]



def run_test(a, b, op, res):
    dataset = ScratchpadDataset(path="datasets/test_scratchpad.pt", operand=op)
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



def run_scratchpad():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    train_dataset = ScratchpadDataset(
        tokenizer=tokenizer,
        train=True,
    )

    eval_dataset = ScratchpadDataset(
        tokenizer=tokenizer,
        train=False,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    args = TrainingArguments(
        output_dir="checkpoints/",
        learning_rate=3e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()



if __name__ == "__main__":
    failed = False
    for a, b, op, res in TESTS:
        failed = True if run_test(a, b, op, res) or failed else False

    run_scratchpad()

    if not failed:
        print("\033[92mAll tests passed!\033[0m")

