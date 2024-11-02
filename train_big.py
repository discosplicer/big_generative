from dataclasses import dataclass
import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from data import DataLoaderLite

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask on the outputs to make it backward-looking only.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, seq length, embedding dimension
        # calculate query, key, value for all heads in batch.
        # nh = num heads, hs = head size, C = nh * hs = channels
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes (T, T) matrix for all queries/keys)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 10257  # number of tokens: 10,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme. Token embedding uses final output weights.
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale standard deviation by inverse square root of 2*num layers.
            # Prevents residuals increasing std.
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # forward through final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Start w/ all candidate parameters.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optimizer groups. Any 2D params will be weight decayed.
        # Generally weights decay and biases don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Use Fused AdamW if available.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("using device: ", device)
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 5000
warmup_steps = 50
def get_lr(it):
    # 1. Linear warmup.
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2. If it > lr_decay_iters, return min learning rate.
    if it > max_steps:
        return min_lr
    # 3. In between, use cosine decay down to min learning rate.
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # starts at 1, goes to 0
    return min_lr + coeff * (max_lr - min_lr)


total_batch_size = 49152
B = 16 # micro batch
T = model.config.block_size
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calc grad accum steps: {grad_accum_steps}")
train_loader = DataLoaderLite(B=B, T=T)



# logits, loss = model(x, y)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
for step in range(max_steps):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # Normalize the loss.
        loss_accum += loss.detach()
        loss.backward()
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    print(f"step {step}, loss: {loss_accum.item():.6f}, norm: {norm:.4f}")

    cutoffs = torch.zeros(1, dtype=torch.bfloat16).to(device)
    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and step % 50 == 0) or (step == max_steps - 1):
        model.eval()
        num_return_sequences = 5
        max_length = 128
        tokens = train_loader.enc.encode("Here is the truth about UFOs:")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # do top-k sampling of 25 tokens on average, but sometimes more or less
                topk_logits, _ = torch.topk(logits, 25, dim=-1)
                # make sure to sample from at least 3
                top3_logits, _ = torch.topk(logits, 3, dim=-1)
                # append the min logit to the list of cutoff logits
                cutoffs = torch.cat((cutoffs, topk_logits.min().reshape(1)))
                # take the average of all cutoff probabilities
                cutoff = torch.minimum(cutoffs.mean(), top3_logits.min().reshape(1))
                # don't sample logits below the cutoff
                logits[logits < cutoff] = -float('Inf')
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                # append to the sequence
                xgen = torch.cat((xgen, idx_next), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = train_loader.enc.decode(tokens)
            print(f"sample {i}: {decoded}")


