from flask import Flask, request, jsonify
from main import load_texts
from tokenizer import SimpleTokenizer
from transformer import Classifier
from constants import block_size
from flask_cors import CORS

import torch

app = Flask(__name__)
CORS(app)

# Initialize the model
texts = load_texts('speechesdataset')
tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data

def load_model():
    model = Classifier(tokenizer.vocab_size)

    # Load the state dictionary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('classifier_model_dict.pth', map_location=device)

    # Set the model to evaluation mode
    model.eval()
    
    return model

model = load_model()

@app.route('/', methods=['GET'])
def home():
    return '<h1>Welcome! The server is running! Send a POST request to /predict please.</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the POST request body
    data = request.json
    text = data['text']

    # Perform inference
    with torch.no_grad():
        wordids = tokenizer.encode(text)
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)
        output, _ = model(input_tensor)
        
    _, predicted = torch.max(output.data, 1)
    
    return jsonify(predicted.tolist())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
