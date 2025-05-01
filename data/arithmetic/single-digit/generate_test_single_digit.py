import random

# Generate all combinations of x + y = z for x, y in 0..9
lines = [f"{x}+{y}={x + y}" for x in range(10) for y in range(10)]

# Shuffle the lines randomly
random.shuffle(lines)

# Write to a text file
with open("random_additions.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")

print("File 'random_additions.txt' created with randomized addition problems.")