from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re, os, sys


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

