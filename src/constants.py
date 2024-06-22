seed = 42

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 1024  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer , OG: 1e-3
n_embd = 64  # Embedding dimension
n_head = 2 # Number of attention heads, OG = 2
n_layer = 4  # Number of transformer layers, OG = 4


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 46  # Output size for the classifier, we have 46 classes
epochs_CLS = 15 # epochs for classifier training