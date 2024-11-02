import tiktoken
import torch
import numpy as np
from datasets import load_dataset

def tokenize(enc, doc):
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        self.reload()
    
    def reload(self):
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-350BT", split="train", streaming=True)
        fw = dataset.shuffle(buffer_size=1000).take(10000)

        self.enc = tiktoken.get_encoding('gpt2')
        # just get the first 10000 + 256 originals
        self.enc._mergeable_ranks = dict(list(self.enc._mergeable_ranks.items())[:10256])
        self.enc._special_tokens = {'<|endoftext|>': 10256}
        self.enc._core_bpe = tiktoken._tiktoken.CoreBPE(self.enc._mergeable_ranks, self.enc._special_tokens, self.enc._pat_str) 
        eot = self.enc._special_tokens['<|endoftext|>'] # end of text token
        tokens = [eot] # the special <|endoftext|> token delimits all documents
        for doc in fw:
            # tokenizes a single document and returns a numpy array of uint16 tokens
            tokens.extend(self.enc.encode_ordinary(doc["text"]))
        # tokens_np = np.array(tokens)
        # assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        # tokens_np_uint16 = tokens_np.astype(np.uint16)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (self.B*self.T)} batches")

        # state
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+(B*T)+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset.
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.reload()
        return x, y