import tiktoken
import torch
import numpy as np
from datasets import load_dataset

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        self.cycles = 0

        self.reload()
    
    def reload(self):
        with open('bge.txt', 'r', encoding='utf-8') as f:
            dataset = f.read()

        self.enc = tiktoken.get_encoding('gpt2')
        # just get the first 10000 + 256 originals
        self.enc._mergeable_ranks = dict(list(self.enc._mergeable_ranks.items())[:10256])
        self.enc._special_tokens = {'<|endoftext|>': 10256}
        self.enc._core_bpe = tiktoken._tiktoken.CoreBPE(self.enc._mergeable_ranks, self.enc._special_tokens, self.enc._pat_str) 
        eot = self.enc._special_tokens['<|endoftext|>'] # end of text token
        tokens = [eot] # the special <|endoftext|> token delimits all documents
        tokens.extend(self.enc.encode_ordinary(dataset))
        # tokens_np = np.array(tokens)
        # assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        # tokens_np_uint16 = tokens_np.astype(np.uint16)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (self.B*self.T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+(B*T)+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset.
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
            self.cycles += 1
        return x, y