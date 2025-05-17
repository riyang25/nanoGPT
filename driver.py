import os
import torch
from eval import eval_model
from model import GPT, GPTConfig

# List of model directories
model_dirs = [
    "out-single-digit-add",
    "out-three-digit-add",
    "out-three-digit-reverse-add-custom-batch",
    "out-three-digit-reverse"
]

# Common settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_file = {
    "out-single-digit-add":"test_single_digit_add.txt",
    "out-three-digit-add":"test_three_digit_add.txt",
    "out-three-digit-reverse-add-custom-batch":"test_three_digit_reverse_add.txt",
    "out-three-digit-reverse":"test_three_digit_reverse_add.txt"
}
dataset = {
    "out-single-digit-add":"arithmetic/single-digit",
    "out-three-digit-add":"arithmetic/three-digit",
    "out-three-digit-reverse-add-custom-batch":"arithmetic/three-digit-reverse",
    "out-three-digit-reverse":"arithmetic/three-digit-reverse"
}
iter_num = 0  # Iteration number for logging (can be adjusted as needed)

# Evaluate each model
for model_dir in model_dirs:
    print(f"Evaluating model in {model_dir}...")

    # Load the model checkpoint
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found in {model_dir}, skipping...")
        continue

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

    # Evaluate the model
    eval_model(model, device, dataset[model_dir], test_file[model_dir], iter_num)

    print(f"Finished evaluating model in {model_dir}.\n")