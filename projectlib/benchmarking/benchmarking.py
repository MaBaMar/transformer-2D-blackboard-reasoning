# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch, re

# MODEL = "gpt2-large"   # fits on 4070 SUPER; use "gpt2" if you prefer
# tok = AutoTokenizer.from_pretrained(MODEL)
# if tok.pad_token_id is None:
#     tok.pad_token = tok.eos_token

# model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")

# FEWSHOT = (
#     "Compute the result. Answer with just the integer.\n"
#     "Q: 3 + 4\nA: 7\n"
#     "Q: 10 + 5\nA: 15\n"
#     "Q: 12 + 1\nA: 13\n"
# )
# ZEROSHOT = ("Compute the result. Answer with just the integer.\n")

# def ask_add(a, b):
#     prompt = FEWSHOT + f"Q: {a} + {b}\nA:"
#     ids = tok(prompt, return_tensors="pt").to(model.device)
#     out = model.generate( **ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,)
#     text = tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
#     m = re.search(r"-?\d+", text)
#     return (int(m.group(0)) if m else None, text.strip())

# for i in range(5):
#     a, b = i*8, i*i*2
#     pred, raw = ask_add(a, b)
#     print(f"Q: {a} + {b} = ?  → model: {raw!r}  |  gt: {a+b}  |  ok? {pred == a+b}")

# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2-medium')
# set_seed(42)
# generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
# llama_zero_shot_math.py

# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import re
# import random
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from projectlib.my_datasets.additions import AdditionDataset


# MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
# TASKS = [(0,0),(12,1),(24,4),(36,9),(48,16)]

# tok = AutoTokenizer.from_pretrained(MODEL)
# if tok.pad_token_id is None:
#     tok.pad_token = tok.eos_token

# pipe = pipeline("text-generation", model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto"),
#                  tokenizer=tok, device_map="auto",)

# def ask(a,b):
#     prompt = f"Compute the result. Answer with just the integer. \n Q: 3 + 4\nA: 7\n Q: 10 + 5\nA: 15\n Q: 12 + 1\nA: 13\n Q: {a}+{b} \nA:"
#     out = pipe(prompt, max_new_tokens=4, do_sample=False,
#                truncation=True, pad_token_id=tok.pad_token_id)[0]["generated_text"]
#     added = out[len(prompt):].strip()
#     m = re.search(r"-?\d+", added)
#     pred = int(m.group(0)) if m else None
#     return added, pred

# for i in range(10):
#     a = random.randint(0, 100)
#     b = random.randint(0, 100)
#     ans_text, pred = ask(a,b)
#     print(f"{a}+{b} -> {ans_text!r}  (pred={pred}, gt={a+b}, ok={pred==a+b})")

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re, random, os, sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)



from projectlib.my_datasets.additions import AdditionDataset
from projectlib.my_datasets.base import GenerationSpec



spec = GenerationSpec(size=100, low=0, high=10000)  
ds = AdditionDataset(train=False, regenerate=True, tokenizer=None, generation_spec=spec)
pairs = list(zip(ds.data, ds.labels))


#MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL = "huggyllama/llama-7b"

tok = AutoTokenizer.from_pretrained(MODEL)


if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

pipe = pipeline("text-generation", model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto"),
    tokenizer=tok, device="cuda",)

def ask(task):
    prompt = (f"Compute the result. Answer with just the integer. \n Q: {task} = ? A:"    )
    out = pipe(prompt, max_new_tokens=20 ,do_sample=False,
                truncation=True, pad_token_id=tok.pad_token_id,)[0]["generated_text"]
    added = out[len(prompt):].strip()
    m = re.search(r"-?\d+", added)
    pred = int(m.group(0)) if m else None
    return added, pred

total = len(pairs)
acc = 0



for input, label in pairs:
    ans_text, pred = ask(input)
    ok = (pred == int(label))
    acc += int(ok)
    print(f"pred={pred}, gt={label}, ok={ok}")

print(f"\nAccuracy: {acc}/{total} = {acc/total:.2%}")

