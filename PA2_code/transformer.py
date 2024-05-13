# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import block_size, n_embd, n_head, n_layer, n_input, n_output, n_hidden

dropout = 0.2

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, decoding=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.decoding = decoding

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_maps):
        B,T,C = x.shape
        
        k = self.key(x)   
        q = self.query(x) 
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        
        if self.decoding:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        wei = F.softmax(wei, dim=-1) 
        
        # wei = self.dropout(wei)
        
        attention_maps.append(wei)
        
        v = self.value(x) 
        
        out = wei @ v 
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, decoding=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, decoding) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_maps):
        out = torch.cat([h(x, attention_maps) for h in self.heads], dim=-1)
        return self.proj(out)
        # return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head=n_head, decoding=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa: MultiHeadAttention = MultiHeadAttention(n_head, head_size, decoding)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-6)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-6)

    def forward(self, x, attention_maps=None):
        x = x + self.sa(self.ln1(x), attention_maps)
        
        # CHANGED FROM x + self.ln2(self.ffwd(x))
        x = self.ln2(x + self.ffwd(x))
        return x

class Classifier(nn.Module):
    def __init__(self, vocab_size, input_size=n_embd, hidden_size=n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer.
        self.fc2 = nn.Linear(hidden_size, n_output)  # Second fully connected layer, outputting three classes.
        self.encoder = Encoder(vocab_size)

    def forward(self, x):
        x, attn_maps = self.encoder(x)
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after the first layer.
        x = self.fc2(x)  # Pass the result to the second layer.
        #  x = F.softmax(x, dim=1)  # Apply Softmax to obtain output probabilities.
        return x, attn_maps
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, decoding=False) for _ in range(n_layer)])

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T)) 
        
        # even_i = torch.arange(0, n_embd, 2).float()
        # odd_i = torch.arange(1, n_embd, 2).float()
        # even_denominator = torch.pow(10000, even_i/n_embd)
        
        # position = torch.arange(block_size, dtype=torch.float).reshape(block_size, 1)
        
        # even_PE = torch.sin(position / even_denominator)
        # odd_PE = torch.cos(position / even_denominator)
        
        # stacked = torch.stack([even_PE, odd_PE], dim=2)
        
        # PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        
        x = tok_emb + pos_emb
        
        attention_maps = []
        
        for block in self.blocks:
           x = block(x, attention_maps) 
        
        x = torch.mean(x, dim=1)
        
        return x, attention_maps
        
class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, decoding=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T))

        
        x = tok_emb + pos_emb 
        
        attention_maps = []
        
        for block in self.blocks:
           x = block(x, attention_maps) 
           
        x = self.ln_f(x) 
        return self.lm_head(x), attention_maps 