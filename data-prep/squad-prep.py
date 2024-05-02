import numpy as np
from datasets import load_dataset
import tiktoken

def prepare_data(examples):
    # Format for completion task: "Context: {context} Question: {question} Answer: {answer_text}"
    prompts_and_answers = [
        "Context: " + context + " Question: " + question + " Answer: " + answer['text'][0]
        for context, question, answer in zip(examples['context'], examples['question'], examples['answers'])
    ]
    return {'prompt': prompts_and_answers}

def encode_data(examples):
    enc = tiktoken.get_encoding("gpt2")
    encodings = [enc.encode_ordinary(prompt) for prompt in examples['prompt']]
    # Add the end of sequence token (EOS) to each encoding
    encodings = [encoding + [enc.eot_token] for encoding in encodings]
    return {'input_ids': encodings}

def save_as_binary(array, filename):
    # Convert list of lists to a numpy array and adjust dtype
    array = [item for sublist in array for item in sublist]
    np_array = np.array(array, dtype=np.uint16)
    # Open file in binary write mode and save the data
    with open(filename, 'wb') as f:
        f.write(np_array.tobytes())

def main():
    # Load SQuAD dataset
    dataset = load_dataset('squad')

    # Prepare the data
    dataset = dataset.map(prepare_data, batched=True)

    # Encode the data
    dataset = dataset.map(encode_data, batched=True, remove_columns=['prompt'])

    # Save encoded data as raw binary files
    for split in ['train', 'validation']:
        save_as_binary(dataset[split]['input_ids'], f'{split}.bin')

if __name__ == "__main__":
    main()
