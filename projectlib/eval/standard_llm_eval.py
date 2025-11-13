from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re, os, sys
import pandas as pd
import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from projectlib.my_datasets.additions import AdditionDataset
from projectlib.my_datasets.base import GenerationSpec
from projectlib.my_datasets.scratchpads import ScratchpadDataset


def get_dataset(task="addition", size=1000, low=0, high=10000):
    if task == "addition":
        spec = GenerationSpec(size=size, low=low, high=high)  
        ds = AdditionDataset(train=False, regenerate=True, tokenizer=None, generation_spec=spec)
        pairs = list(zip(ds.data, ds.labels))
    elif task == "scratch_pad":
        spec = GenerationSpec(size=size, low=low, high=high)  
        ds = ScratchpadDataset(train=False, regenerate=True, tokenizer=None, generation_spec=spec)
        pairs = list(zip(ds.data, ds.labels))
    return pairs

def setup_model(model, device=-1):
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pipe = pipeline("text-generation", model=AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto"),
        tokenizer=tok, device=device)
    return pipe, tok

def ask(input, task, pipe, tok):
    if task == "addition":
        prompt = (f"Compute the sum. Answer with just the integer. \n Q: {input} = ? A:"    )
        out = pipe(prompt, max_new_tokens=30 ,do_sample=False,
                    truncation=True, pad_token_id=tok.pad_token_id,)[0]["generated_text"]
        added = out[len(prompt):].strip()
        m = re.search(r"-?\d+", added)
        if m:
            pred = int(m.group(0))
        else:
            return None
    elif task == "scratch_pad":
        prompt = (f"You are a calculator. Show your work between <scratch> and </scratch>. Then output a single line: \"Result: <number>\" and stop. Your Task including an Example: {input}")
        out = pipe(prompt, max_new_tokens=200 ,do_sample=False,
                    truncation=True, pad_token_id=tok.pad_token_id,)[0]["generated_text"]
        added = out[len(prompt):].strip()
        print(f"Scratchpad output: {added}")
        m = re.findall(r"Result:\s*([0-9 ]+)", added)
        if m:
            raw = m[0].strip()
        else:
            return None
        pred = int(raw.replace(" ", ""))
    return pred

def extract_label_number(label: str) -> int | None:
    """
    Extract the final numeric result from a dataset label string.
    Returns an int or None if no match.
    """
    # Look for the last 'Result:' followed by digits and spaces
    m = re.findall(r"Result:\s*([0-9 ]+)", label)
    raw = m[-1].strip().replace(" ", "")
    label = int(raw) if raw.isdigit() else None
    return label

def ask_batch(task, inputs, pipe, tok, batch_size=8):
    if task == "addition":
        prompts = [f"Compute the sum. Answer with just the integer.\nQ: {inp} = ?\nA:" for inp in inputs]
        outs = pipe(prompts, max_new_tokens=30, do_sample=False, batch_size=batch_size,
                    truncation=True, pad_token_id=tok.pad_token_id)
        preds = []
        for out, prompt in zip(outs, prompts):
            added = out[0]["generated_text"][len(prompt):].strip()
            m = re.search(r"-?\d+", added)
            preds.append(int(m.group(0)) if m else None)
        return preds

    elif task == "scratch_pad":
        outs = pipe(inputs, max_new_tokens=200, do_sample=False, batch_size=batch_size,
                    truncation=True, pad_token_id=tok.pad_token_id)
        preds = []
        for out, prompt in zip(outs, inputs):
            added = out[0]["generated_text"][len(prompt):].strip()
            m = re.findall(r"Result:\s*([0-9 ]+)", added)
            if not m:
                preds.append(None)
                continue
            raw = m[0].strip().replace(" ", "")
            preds.append(int(raw) if raw.isdigit() else None)
        return preds
    
def main():
    df = pd.DataFrame(columns =["model", "task", "max_summand", "accuracy"])
    model_lst = ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "huggyllama/llama-7b")
    settings =  [(100, 0,10), (100, 10, 100), (100, 100, 1000), (100, 1000, 10000),  (100, 10000, 100000)]
    device = 0 if torch.cuda.is_available() else -1 # 0 for GPU, -1 for CPU

    for model in model_lst:
        pipe, tok = setup_model(model, device)
        for task in ("addition","scratch_pad"): 
            for size, low, high in settings:
                pairs = get_dataset(task=task, size=size, low=low, high=high)
                correct = 0
                print(f"Evaluating model={model}, task={task}, max_summand={high}, size={size}")
                batch_size = 8
                for i in tqdm.tqdm(range(0, len(pairs), batch_size)):
                    batch = pairs[i:i+batch_size]
                    inputs, labels = zip(*batch)
                    inputs = [str(x) for x in inputs]      


                    if task == "scratch_pad":
                        labels = [extract_label_number(lbl) for lbl in labels]
                    else:
                        labels = [int(lbl) for lbl in labels]

                    preds = ask_batch(task, inputs, pipe, tok, batch_size)
                    for pred, label in zip(preds, labels):
                        ok = (pred is not None and label is not None and int(pred) == int(label))
                        correct += int(ok)

                acc = correct/size
                df.loc[len(df)] = [model, task, high, acc]
                df.to_csv("projectlib/eval/backup.csv", index=False)
    df.to_csv("projectlib/eval/standard_llm_eval_results.csv", index=False)

if __name__ == "__main__":
    main()