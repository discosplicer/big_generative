import tiktoken
import torch
import numpy as np
from datasets import load_dataset

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
total_batch_size = 49152

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
    def __init__(self, B, T, model, quantile, data='fineweb'):
        self.B = B
        self.T = T
        self.model = model
        self.quantile = quantile
        self.data = data

        

        self.reload()
    
    def reload(self):
        dataset = load_dataset(f"HuggingFaceFW/{self.data}", name="sample-350BT", split="train", streaming=True)
        fw = dataset.shuffle(buffer_size=1000).take(10000)

        self.cycle_trigger = False

        self.enc = tiktoken.get_encoding('gpt2')
        # just get the first 10000 + 256 originals
        self.enc._mergeable_ranks = dict(list(self.enc._mergeable_ranks.items())[:10256])
        self.enc._special_tokens = {'<|endoftext|>': 10256}
        self.enc._core_bpe = tiktoken._tiktoken.CoreBPE(self.enc._mergeable_ranks, self.enc._special_tokens, self.enc._pat_str) 
        eot = self.enc._special_tokens['<|endoftext|>'] # end of text token
        tokens = [eot] # the special <|endoftext|> token delimits all documents
        dt = []  # list of document tokens
        dl = []  # list of document losses
        for doc in fw:
            # score each document
            doc_tokens = self.enc.encode_ordinary(doc["text"])
            dt.append(doc_tokens)
            torch_tokens = torch.tensor(doc_tokens, dtype=torch.long)
            if (self.T + 2) > len(doc_tokens):
                # short document, don't bother
                dl.extend([100.0])
            else:
                start_pos = np.random.choice(len(doc_tokens) - self.T - 1)
                buf = torch_tokens[start_pos : start_pos + self.T + 1]
                x = (buf[:-1]).view(1, self.T) # inputs
                y = (buf[1:]).view(1, self.T) # targets
                x, y = x.to(device), y.to(device)
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = self.model(x, y) # (B, T, vocab_size)
                dl.extend([loss.item()])
        dl = np.array(dl)
        threshold = np.quantile(dl, self.quantile)
        top_indices = np.where(dl <= threshold)[0]
        for index in top_indices:
            tokens.extend(dt[index])
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
        grad_accum_steps = total_batch_size // (B * T)
        buf = self.tokens[self.current_position : self.current_position+(B*T)+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset.
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.reload()
        return x, y