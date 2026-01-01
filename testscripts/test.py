from projectlib.my_datasets import ScratchpadDataset, GenerationSpec, Split

spec = GenerationSpec(0, 100, 1, 1, 1)
data_eval = ScratchpadDataset(spec, split=Split.EVAL)
data_test = ScratchpadDataset(spec, split=Split.TEST)
data_train = ScratchpadDataset(spec, split=Split.TRAIN)

print("-" * 10 + "Eval" + "-" * 10 + "\n")
for d in data_eval:
    print("input:\n" + d["input"])
    print("label:\n" + d["label"])

print("\n" + "-" * 10 + "Test" + "-" * 10 + "\n")
for d in data_test:
    print("input:\n" + d["input"])
    print("label:\n" + d["label"])

print("\n" + "-" * 10 + "Train" + "-" * 10 + "\n")
for d in data_train:
    print("input:\n" + d["input"])
    print("label:\n" + d["label"])
