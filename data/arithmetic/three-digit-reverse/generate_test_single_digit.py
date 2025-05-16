import random
import os
# Generate all combinations of x + y = z for x, y in 0..9
lines = [f"{x}+{y}={str(x + y)[::-1]}" for x in range(1000) for y in range(1000)]

# Shuffle the lines randomly
random.shuffle(lines)

# Write to a text file
with open("./data/arithmetic/three-digit-reverse/test_three_digit_reverse_add.txt", "w") as f:
    for i in range(1000):
        f.write(lines[i] + "\n")

print("File 'random_additions.txt' created with randomized addition problems.")