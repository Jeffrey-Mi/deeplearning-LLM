# Used https://github.com/karpathy/nanoGPT as reference for implementation of shards

import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

# Setup number of processes
encoder = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the dataset with specified number of processes
    dataset = load_dataset("openwebtext", num_proc=num_processes)

    # Define tokenization process
    def tokenize(example):
        token_ids = encoder.encode_ordinary(example['text'])
        token_ids.append(encoder.eot_token)
        return {'ids': token_ids, 'len': len(token_ids)}

    # Apply tokenization to the dataset
    tokenized_data = dataset.map(
        tokenize,
        remove_columns=['text'],
        num_proc=8,
    )

    # Write tokenized data to binary files
    for split_name, data in tokenized_data.items():
        total_length = np.sum(data['len'], dtype=np.uint64)
        data_array = np.memmap('train.bin', dtype=np.uint16, mode='w+', shape=(total_length,))
        
        index = 0
        # Create and write batches to the memory-mapped array
        for batch_index in tqdm(range(1024), desc=f'Writing to train.bin'):
            shard = data.shard(num_shards=1024, index=batch_index, contiguous=True).with_format('numpy')
            batch_data = np.concatenate(shard['ids'])
            data_array[index: index + len(batch_data)] = batch_data
            index += len(batch_data)
        data_array.flush()
