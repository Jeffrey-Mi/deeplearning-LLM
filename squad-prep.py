import torch
from torch import nn, Tensor
# import torchsummary
from tqdm.auto import tqdm
import numpy as np

import pandas as pd
from tqdm.notebook import tqdm as blue_tqdm
import json

import math
from typing import Optional, List
import wandb

import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

config = {
    'context_len': 512,
    'vocab_size': 50304,
    'num_heads': 8,
    'd_model': 512
}

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        # regularization
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = 0.1
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config['context_len'], config['context_len']))
                                        .view(1, 1, config['context_len'], config['context_len']))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedFoward(nn.Module):

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_seq_len= config['context_len'], dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x

class Decoder(nn.Module):

    def __init__(self, n_layer=5):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['d_model'])
        self.positional_encoding = PositionalEncoding(config['d_model'])
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd=config['d_model'], n_head=config['num_heads']) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(config['d_model']) # final layer norm
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = self.positional_encoding(tok_emb) # (T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config['context_len']:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model   = Decoder(
    n_layer=5
).to(DEVICE)


print(model)



checkpoint = torch.load('artifacts/model:v59/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

import json
import tiktoken
enc = tiktoken.get_encoding("gpt2")

def generate_answer(context, question):
    input_text = "Context: " + context
    input_text += " Question: " + question
    input_text += " Answer: "
    prompt = torch.tensor([enc.encode(input_text)]).to(DEVICE)

    answer_start = len(prompt[0])
    model.eval()
    model_out = model.generate(prompt, 50)[0].tolist()
    try:
      answer_end = model_out.index(enc.eot_token)
    except:
      answer_end = len(model_out)

    answer_enc = model_out[answer_start:answer_end]
    return enc.decode(answer_enc)
  

def main():
    # Load the SQuAD development dataset.
    with open('dev-v2.0.json', 'r') as file:
        squad_data = json.load(file)

    answers = {}
    
    # Process each article, paragraph, and question.
    for article in tqdm(squad_data['data']):
        for paragraph in tqdm(article['paragraphs']):
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                id = qa['id']
                gen_ans = generate_answer(context, question)
                answers[id] = gen_ans
            
    
    # Output the modified dataset with generated answers.
    with open('dev-with-generated-answers.json', 'w') as outfile:
        json.dump(answers, outfile, indent=4)

main()
