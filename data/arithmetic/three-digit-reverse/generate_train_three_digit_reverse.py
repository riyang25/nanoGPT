import random

def generate_random_additions(filename="./data/arithmetic/three-digit-reverse/train_three_digit_reverse_add.txt", num_lines=1000):
    with open(filename, "w") as f:
        for _ in range(num_lines):
            x = random.randint(0, 1000)
            y = random.randint(0, 1000)
            sumreverse = str(x + y).zfill(4)[::-1]
            x = str(x).zfill(3)
            y = str(y).zfill(3)
            f.write(f"{x}+{y}={sumreverse}\n")

    print(f"File '{filename}' created with {num_lines} random addition problems.")

# Example usage:
generate_random_additions(num_lines=100000)
generate_random_additions(filename="./data/arithmetic/three-digit-reverse/test_three_digit_reverse_add.txt", num_lines=1000)
