import random

def generate_random_additions(filename="./data/arithmetic/three-digit/train_three_digit_add.txt", num_lines=1000):
    with open(filename, "w") as f:
        for _ in range(num_lines):
            x = random.randint(0, 1000)
            y = random.randint(0, 1000)
            f.write(f"{x}+{y}={x + y}\n")

    print(f"File '{filename}' created with {num_lines} random addition problems.")

# Example usage:
generate_random_additions(num_lines=10000)
