# Teaching Transformers Arithmetic
Repository: (https://github.com/riyang25/nanoGPT)
A driver file, `driver.py` can be run to reproduce evaluation results for the models.

Models trained in the course of this project can be found in folders `out-single-digit-add`, `out-three-digit-add`, `out-three-digit-reverse-add`, and `out-three-digit-reverse-add-custom-batch`.

In order to train the single digit addition model, run
`python3 train.py config/single_digit-add.py`.

In order to train the three digit addition model, run
`python3 train.py config/three_digit_add.py`.

In order to train the reversed three digit addition model with default data loading, run
`python3 train.py config/three_digit_add_reverse.py --custom_batch=False --out-dir=out-three-digit-add-reverse`.

In order to train it with the custom data loading, run
`python3 train.py config/three_digit_add_reverse.py --custom_batch=True --outdir=out-three-digit-add-reverse-custom-batch`.