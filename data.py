import tiktoken
import torch

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        # at init load tokens from disk
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        # just get the first 1000 + 256 originals
        enc._mergeable_ranks = dict(list(enc._mergeable_ranks.items())[:1256])
        enc._core_bpe = tiktoken._tiktoken.CoreBPE(enc._mergeable_ranks, enc._special_tokens, enc._pat_str) 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

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
        return x, y