# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-single-digit-add'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'single-digit-add'
wandb_run_name = 'mini'

dataset = 'arithmetic/single-digit'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 32 # context of up to n previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

test_file = 'test_single_digit_add.txt'

# on macbook also add
device = 'mps'  # run on cpu only
compile = False # do not torch compile the model
