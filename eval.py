import torch
from model import GPT, GPTConfig
from contextlib import nullcontext
import pickle
import os
import random
import tiktoken

def eval_model(model, device, dataset, test_file, iter_num, log_file=None, loss=None):
    model.eval()
    data_dir = os.path.join('data', dataset)
    test_file = os.path.join(data_dir, test_file)
    # Load the test cases
    with open(test_file, "r") as f:
        test_cases = [line.strip() for line in f.readlines()]

    random.shuffle(test_cases)

    # Initialize tokenizer
    digits = [str(i) for i in range(10)]
    special = ['+','=','\n']
    vocab = digits+special

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(vocab) }
    itos = { i:ch for i,ch in enumerate(vocab) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Evaluate the model
    correct = 0
    total = 0
    correct_chars = 0
    total_chars = 0
    for case in test_cases:
        prompt, expected = case.split("=")
        prompt = prompt + "="  # Ensure the prompt ends with "="
        expected = expected.strip()

        # Tokenize input
        input_ids = encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        # Generate output
        with torch.no_grad():
            output = model.generate(input_tensor, max_new_tokens=5)  # Adjust max_new_tokens if needed
        output_text = decode(output[0].tolist())

        # Extract the model's prediction
        prediction = output_text[len(prompt):].split("\n")[0].strip()

        # Compare with the expected result
        if prediction == expected:
            correct += 1
        # else:
            # print(f"Test case: {case}")
            # print(f"Expected: {expected}, Predicted: {prediction}")
        total += 1
        for i in range(len(prediction)):
            if i < len(expected) and prediction[i] == expected[i]:
                correct_chars += 1
            total_chars += 1

    # Calculate accuracy
    accuracy = correct / total * 100
    char_accuracy = correct_chars / total_chars * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Character Accuracy: {char_accuracy:.2f}%")
    if log_file:
        log_file = os.path.join('./training-logs', log_file)
        with open(log_file, "a") as f:
            f.write(f"Iteration {iter_num}:\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            f.write(f"Character Accuracy: {char_accuracy:.2f}%\n")
    return accuracy, char_accuracy

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
seed = 1337 # for reproducibility
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
dataset = None
test_file = None
log_file = None
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
# TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
# ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


eval_model(model, device, dataset, test_file, 0, log_file=log_file)