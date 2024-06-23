from fastapi import FastAPI, Request
from pydantic import BaseModel
from main import load_texts
from tokenizer import SimpleTokenizer
from transformer import Classifier
from constants import block_size
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

block_size = 1024

model = None
tokenizer = None

def initialize():
    global model, tokenizer
    
    if not tokenizer:
        texts = [text.split('\t', 1)[1] for text in load_texts('train.tsv')]
        tokenizer = SimpleTokenizer(' '.join(texts))
        
    if not model:
        model = Classifier(tokenizer.vocab_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load('all_pres_classifier_model_dict.pth', map_location=device))
        model.to(device)
        model.eval()

class TextInput(BaseModel):
    text: str

@app.get("/")
async def home():
    return {"message": "Welcome! The server is running! Send a POST request to /predict please."}

@app.post("/predict")
async def predict(request: TextInput):
    global model, tokenizer
    if model is None or tokenizer is None:
        initialize()
        
    # Get the text from the POST request body
    text = request.text

    # Perform inference
    with torch.no_grad():
        wordids = tokenizer.encode(text)
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)
        
        output, _ = model(input_tensor)
        
    _, predicted = torch.max(output.data, 1)
    
    return {"predicted": predicted.tolist()}

