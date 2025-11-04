"""
Function for dataset generation. Feel free to define new functions here and to use them in the code.
Please regularly push updated versions of the library, so others can use the same functionality.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

from my_datasets.additions import AdditionDataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    train_dataset = AdditionDataset(
        tokenizer=tokenizer,
        train=True,
    )

    eval_dataset = AdditionDataset(
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