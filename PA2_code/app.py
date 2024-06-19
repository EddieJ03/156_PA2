# app.py
import pickle
from flask import Flask, request, jsonify
from transformer import Classifier
from transformers import GPT2Tokenizer
import torch

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    return '<h1>It works!</h1>'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
