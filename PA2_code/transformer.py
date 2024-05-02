# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import block_size, n_embd, n_head, n_layer, n_input, n_output, n_hidden

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, decoding=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.decoding = decoding

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_maps):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        if self.decoding:
            # print("mask??")
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # wei = self.dropout(wei)
        
        attention_maps.append(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, decoding=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, decoding) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_maps):
        out = torch.cat([h(x, attention_maps) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 8*n_embd),
            nn.GELU(),
            nn.Linear(8*n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head=n_head, decoding=False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa: MultiHeadAttention = MultiHeadAttention(n_head, head_size, decoding)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_maps=None):
        x = x + self.sa(self.ln1(x), attention_maps)
        x = x + self.ffwd(self.ln2(x))
        return x

class Classifier(nn.Module):
    def __init__(self, input_size=n_embd, hidden_size=n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer.
        self.fc2 = nn.Linear(hidden_size, n_output)  # Second fully connected layer, outputting two classes.

    # Define the forward pass of the neural network.
    # x: The input tensor.
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after the first layer.
        x = self.fc2(x)  # Pass the result to the second layer.
        x = F.softmax(x, dim=1)  # Apply Softmax to obtain output probabilities.
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, decoding=False) for _ in range(n_layer)])
        # self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.classifier = Classifier(input_size=n_embd, hidden_size=n_hidden)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        attention_maps = []
        
        for block in self.blocks:
           x = block(x, attention_maps) # (B,T,C)
        
        # x = self.ln_f(x) # (B,T,C)
        
        x = torch.mean(x, dim=1)
        
        return self.classifier(x), attention_maps
        
class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, decoding=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        
        x = tok_emb + pos_emb # (B,T,C)
        
        attention_maps = []
        
        for block in self.blocks:
           x = block(x, attention_maps) # (B,T,C)
           
        x = self.ln_f(x) # (B,T,C)
        return self.lm_head(x), attention_maps # (B,T,vocab_size)