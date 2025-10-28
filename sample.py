"""
sample.py - importable function to generate text from trained nanoGPT
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# Load model once
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# Setup device context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device=='cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# Load meta.pkl
meta_path = '/content/nanoGPT/data/shakespeare_char/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# -----------------------------------------------------------------------------
def sample(prompt, max_new_tokens=200, temperature=0.8, top_k=50, seed=1337):
    torch.manual_seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(y[0].tolist())

# -----------------------------------------------------------------------------
# Optional: run as script
if __name__ == "__main__":
    context = "Hamlet: "
    output = sample(context)
    print(output)
