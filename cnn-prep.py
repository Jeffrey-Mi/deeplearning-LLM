import numpy as np
from datasets import load_dataset
import tiktoken
enc = tiktoken.get_encoding("gpt2")

def prepare_data(examples):
    for article, summary in zip(examples['article'], examples['highlights']):
        article = enc.encode_ordinary(article)[:504]
        article = enc.encode_ordinary("(Article): ") + article + enc.encode_ordinary(" (Summary): ")
        article = [enc.eot_token]*(512 - len(article)) + article
        zamps.append(article)
    prompts_and_summaries = [
        "(Article): " + article + " (Summary): " + summary
        for article, summary in zip(examples['article'], examples['highlights'])
    ]
    return {'prompt': prompts_and_summaries}

def encode_data(examples):
    encodings = [enc.encode_ordinary(prompt) for prompt in examples['prompt']]
    encodings = [encoding + [enc.eot_token] for encoding in encodings]
    return {'input_ids': encodings}

def save_as_binary(array, filename):
    array = [item for sublist in array for item in sublist]
    np_array = np.array(array, dtype=np.uint16)
    with open(filename, 'wb') as f:
        f.write(np_array.tobytes())

def main():
    dataset = load_dataset('cnn_dailymail', '3.0.0')

    dataset = dataset.map(prepare_data, batched=True)

    dataset = dataset.map(encode_data, batched=True, remove_columns=['prompt'])

    for split in ['train', 'validation']:
        save_as_binary(dataset[split]['input_ids'], f'cnn_{split}.bin')

if __name__ == "__main__":
    main()
