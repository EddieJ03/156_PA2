import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

import os

from utilities import Utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from constants import seed, batch_size, block_size, learning_rate, n_embd, n_head, n_layer, n_input, n_output, n_hidden, epochs_CLS

from transformer import Classifier, Decoder, DecoderEC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels


def compute_perplexity(decoderLMmodel: Decoder, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        logits, _ = decoderLMmodel(X) # your model should be computing the cross entropy loss
        B, T, C = logits.shape
        
        logits = logits.view(B*T, C)
        targets = Y.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters: 
            break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def compute_classifier_accuracy(classifier: Classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def train_epoch(data_loader, model, optimizer):
    # size = len(data_loader.dataset)
    
    num_batches = len(data_loader)
    model.train()
    train_loss, total_correct, total_samples = 0, 0, 0
    
    for batch, (X, Y) in enumerate(data_loader):
        #  X = X.float()
        # Compute prediction error
        # print('----------------------------------------------------')
        pred, _ = model(X)
        
        _, predicted = torch.max(pred.data, 1)
        total_correct += (predicted == Y).sum().item()
        total_samples += Y.size(0)
        
        # print("pred: ", pred.shape)
        # print("predicted", predicted.shape)
        # print("y", Y.shape)
        # print('----------------------------------------------------')
        
        loss = F.cross_entropy(input=pred, target=Y)
        
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = (100 * total_correct / total_samples)
    return accuracy, average_train_loss


# ------------------------------Classifier Code---------------------------------- #
def run_classifier():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    
    classifier_model = Classifier(tokenizer.vocab_size)
    
    total_params = sum(p.numel() for p in classifier_model.parameters())
    print("Total number of parameters:", total_params)

    # Adam optimizer
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)
    
     # for the classification task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        train_accuracy, train_loss = train_epoch(train_CLS_loader, classifier_model, optimizer)
        print(f'Epoch #{epoch+1}: \t train accuracy {train_accuracy:.3f}\t train loss {train_loss:.3f}\t test accuracy {compute_classifier_accuracy(classifier_model, test_CLS_loader):.3f}')
    
    # SANITY CHECK
    print("------------------------SANITY CHECK ENCODER------------------------")   
    tokenizer = SimpleTokenizer('The man who passes the sentence should swing the sword. If you would take a man\'s life') # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)   
    
    ec = Classifier(tokenizer.vocab_size)
    u = Utilities(tokenizer, ec)
    
    u.sanity_check('The man who passes the sentence should swing the sword. If you would take a man\'s life', block_size)
    
# ------------------------------Classifier Code---------------------------------- #
    
    
# ------------------------------Decoder Code---------------------------------- #     
def run_decoder():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)   

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    
    decoder = Decoder(tokenizer.vocab_size)
    
    total_params = sum(p.numel() for p in decoder.parameters())
    print("Total number of parameters:", total_params)
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters: # stop after 500 batches
            break
        xb, yb = xb.to(device), yb.to(device)
        
        if (i+1)%100 == 0:
            print("Train Data Perplexity At Iteration ", (i+1), compute_perplexity(decoder, train_LM_loader))
        
        # LM training code here
        
        # evaluate the loss
        logits, _ = decoder(xb)
        B, T, C = logits.shape
        
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate perplexity
    files = ['speechesdataset/test_LM_hbush.tsv', 'speechesdataset/test_LM_obama.txt', 'speechesdataset/test_LM_wbush.txt']
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, data,  block_size)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)
        
        print(file, compute_perplexity(decoder, test_LM_loader))
    
    print("------------------------SANITY CHECK DECODER------------------------")
    tokenizer = SimpleTokenizer('The man who passes the sentence should swing the sword. If you would take a man\'s life') # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)   
    
    ec = Decoder(tokenizer.vocab_size)
    u = Utilities(tokenizer, ec)
    
    u.sanity_check('The man who passes the sentence should swing the sword. If you would take a man\'s life', block_size)
        
# ------------------------------Decoder Code---------------------------------- #  


def run_sanity_check_encoder():
    print("Loading data and creating tokenizer ...")
    tokenizer = SimpleTokenizer('The man who passes the sentence should swing the sword. If you would take a man\'s life') # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)   
    
    ec = Classifier(tokenizer.vocab_size)
    u = Utilities(tokenizer, ec)
    
    u.sanity_check('The man who passes the sentence should swing the sword. If you would take a man\'s life', block_size)
    
    
def run_sanity_check_decoder():
    print("Loading data and creating tokenizer ...")
    tokenizer = SimpleTokenizer('The man who passes the sentence should swing the sword. If you would take a man\'s life') # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)   
    
    ec = Decoder(tokenizer.vocab_size)
    u = Utilities(tokenizer, ec)
    
    u.sanity_check('The man who passes the sentence should swing the sword. If you would take a man\'s life', block_size)


def run_ec_decoder():
    print("EXTRA CREDIT:")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)   

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    
    decoder = DecoderEC(tokenizer.vocab_size, 4, 6)
    
    total_params = sum(p.numel() for p in decoder.parameters())
    print("Total number of parameters:", total_params)
    
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters: # stop after 500 batches
            break
        xb, yb = xb.to(device), yb.to(device)
        
        if (i+1)%100 == 0:
            print("Train Data Perplexity At Iteration ", (i+1), compute_perplexity(decoder, train_LM_loader))
        
        # LM training code here
        
        # evaluate the loss
        logits, _ = decoder(xb)
        B, T, C = logits.shape
        
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate perplexity
    files = ['speechesdataset/test_LM_hbush.tsv', 'speechesdataset/test_LM_obama.txt', 'speechesdataset/test_LM_wbush.txt']
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, data,  block_size)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)
        
        for i in range(5):
            print("Iteration", (i+1)*100, file, compute_perplexity(decoder, test_LM_loader))
            
# ------------------------------MAIN---------------------------------- #  
def main():
    parser = argparse.ArgumentParser(description="Run classifier or decoder")
    parser.add_argument("-mode", choices=["c", "d", "sc", "sd", "ecd"], help="Choose mode: 'c' for classifier, 'd' for decoder, 'sc' for sanity checking classifier, 'sd' for sanity checking decoder, 'ecd' for decoder EC")
    args = parser.parse_args()

    if args.mode == "c":
        run_classifier()
    elif args.mode == "d":
        run_decoder()
    elif args.mode == 'sc':
        run_sanity_check_encoder()
    elif args.mode == 'sd':
        run_sanity_check_decoder()
    elif args.mode == 'ecd':
        run_ec_decoder()
    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()
