import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Hyperparameters

# Network Tuning parameters
batch_size = 64 # 
block_size = 256 # Context length for predictions
n_embd = 384 # length of embedding dimension
n_head = 6 # number of heads
n_layer = 6 # number of layers
dropout_freq = 0.2

# Training Parameters
lr = 3e-4 # Learning Rate
max_epochs = 5000 #
evaluation_interval = 500 #
evaluation_iterations = 200

torch.manual_seed(4269)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CausalMultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        # key, query, and value for each head
        self.qkv_atten = nn.Linear(n_embd, 3*n_embd)
        
        # output projection tensor
        self.projection = nn.Linear(n_embd, n_embd)
        
        # Regularization layers
        self.atten_dropout = nn.Dropout(dropout_freq)
        self.resid_dropout = nn.Dropout(dropout_freq)
        
        # members
        self.num_head = num_heads
        self.num_embd = n_embd
        self.dropout_freq = dropout_freq
        
        # flash attention is much faster but requires PyTorch >= 2.0
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # create a causal mask
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))
    
    def forward(self, x):
        # batch size, sequence length, num_embd dimensionality
        B, T, C = x.size()
        
        # split qkv into each seperate Linear layer
        q, k, v = self.qkv_atten(x).split(self.num_embd, dim=2)
        
        # Reshape k, q, v to be in form (B, num heads, T, head size)
        k = k.view(B, T, self.num_head, C // self.num_head).transpose(1, 2)
        q = q.view(B, T, self.num_head, C // self.num_head).transpose(1, 2)
        v = v.view(B, T, self.num_head, C // self.num_head).transpose(1, 2)
        
        # Dot product attention
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_freq, is_causal=True)
        else:
            weights = (q @ k.transpose(-2, -1)) * (1.0/ math.sqrt(k.size(-1))) # (B, num heads, T, T)
            weights = weights.masked_fill(self.mask[:,:,T,T] == 0, float('-inf'))
            weights = F.softmax(weights, dim=-1)
            weights = self.atten_dropout(weights)
            y = weights @ v # (B, num heads, T, head size)
        
        # reassemble the heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.projection(y))
        return y


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_freq),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block"""
    
    def  __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.causal_attention = CausalMultiHeadAttention(n_head, head_size)
        self.mlp = MLP(n_embd)
        self.lnorm1 = nn.LayerNorm(n_embd)
        self.lnorm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # x + network because residual connection makes backprop quicker early in training
        x = x + self.causal_attention(self.lnorm1(x))
        x = x + self.mlp(self.lnorm2(x))
        return x


class GPT(nn.Module):
    """Generative Pre-trained Transformer"""
    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.final_lnorm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights to zero mean Normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_lnorm(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probabilities, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx