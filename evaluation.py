import pandas as pd
import argparse
import re
import string
import numpy as np
# Extract the last letter (A-Z) between triple hashes, e.g., ###A###
def extract_letter(response):
    matches = re.findall(r'###([A-Z])###', str(response))
    return matches[-1] if matches else None

# Extract the last integer between triple hashes, e.g., ###2###
def extract_number(response):
    matches = re.findall(r'###([0-9]+)###', str(response))
    return int(matches[-1]) if matches else None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--response_field', type=str, required=True)
    parser.add_argument('--task_type', type=str, required=True, choices=['mcq_numbers', 'mcq_letters'])
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    ground_truth = df['gold_answer'].tolist()
    if args.task_type == 'mcq_letters':
        letter2num = {l: i for i, l in enumerate(string.ascii_uppercase)}
        predictions = df[args.response_field].apply(extract_letter).tolist()
        predictions = ['None' if p is None else letter2num[p] for p in predictions]
        correctness = [p == g for p, g in zip(predictions, ground_truth)]
        accuracy = sum(correctness) / len(correctness)
        # Bootstrap variance
        n = len(correctness)
        boot_accuracies = [np.mean(np.random.choice(correctness, n, replace=True)) for _ in range(1000)]
        variance = np.var(boot_accuracies)
        print(f'Accuracy: {accuracy}')
        print(f'Bootstrap Variance: {variance}')
    elif args.task_type == 'mcq_numbers':
        predictions = [p-1 for p in df[args.response_field].apply(extract_number)]
        predictions = [-1 if p is None else p for p in predictions]
        correctness = [p == g for p, g in zip(predictions, ground_truth)]
        accuracy = sum(correctness) / len(correctness)
        # Bootstrap variance
        n = len(correctness)
        boot_accuracies = [np.mean(np.random.choice(correctness, n, replace=True)) for _ in range(1000)]
        variance = np.var(boot_accuracies)
        print(f'Accuracy: {accuracy}')
        print(f'Bootstrap Variance: {variance}')
    else:
        raise ValueError(f'Task type {args.task_type} not supported')


