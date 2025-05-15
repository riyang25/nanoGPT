import torch
from model import GPT
import os
import random

def eval_model(model, device, dataset, test_file):
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
    return accuracy, char_accuracy