import random

# Generate all combinations of x + y = z for x, y in 0..9
lines = [f"{x}+{y}={x + y}" for x in range(1000) for y in range(1000)]

# Shuffle the lines randomly
random.shuffle(lines)

# Write to a text file
with open("./data/arithmetic/three-digit/test_three_digit_add.txt", "w") as f:
    for i in range(1000):
        f.write(lines[i] + "\n")

print("File 'random_additions.txt' created with randomized addition problems.")