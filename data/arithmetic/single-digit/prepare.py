"""

"""
import os
import pickle
import requests
import numpy as np

train_file_path = os.path.join(os.path.dirname(__file__), 'train_single_digit_add.txt')
test_file_path = os.path.join(os.path.dirname(__file__), 'test_single_digit_add.txt')

with open(train_file_path, 'r') as f:
    train = f.read()
print(f"length of dataset in characters: {len(train):,}")

with open(test_file_path, 'r') as f:
    test = f.read()

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

# create the train and test splits


# encode both to integers
train_ids = encode(train)
val_ids = encode(test)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': len(vocab),
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(decode(encode('2+3=5')))