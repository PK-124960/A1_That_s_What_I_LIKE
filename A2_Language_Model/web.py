import streamlit as st
import torch
from torch.nn.functional import softmax
from torchtext.data.utils import get_tokenizer
from datasets import load_dataset
import torch.nn as nn

# Load model and vocab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset and tokenizer
tokenizer = get_tokenizer("basic_english")
dataset = load_dataset("KaungHtetCho/Harry_Potter_LSTM")

# Tokenize data
def tokenize_data(example):
    return {"tokens": tokenizer(example["text"])}

tokenized_dataset = dataset.map(tokenize_data, remove_columns=["text"])

# Build vocabulary
from torchtext.vocab import build_vocab_from_iterator

vocab = build_vocab_from_iterator(tokenized_dataset["train"]["tokens"], min_freq=3)
vocab.insert_token("<unk>", 0)
vocab.insert_token("<eos>", 1)
vocab.set_default_index(vocab["<unk>"])

# Define model architecture (must match training)
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, 1024).to(device),  # Hidden state
                torch.zeros(2, batch_size, 1024).to(device))  # Cell state

    def forward(self, src, hidden):
        embedded = self.embedding(src)
        output, hidden = self.lstm(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden

# Load model
model = LSTMLanguageModel(len(vocab), emb_dim=1024, hid_dim=1024, num_layers=2, dropout_rate=0.65).to(device)
model.load_state_dict(torch.load("best-val-lstm_lm.pt", map_location=device))
model.eval()

# Define text generation function
def generate_text(prompt, max_seq_len, temperature):
    tokens = tokenizer(prompt)
    indices = [vocab[token] for token in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = softmax(prediction[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == vocab["<eos>"]:
                break
            indices.append(next_token)

    itos = vocab.get_itos()
    return " ".join(itos[idx] for idx in indices)

# Streamlit UI
st.title("LSTM Language Model Demo")
st.write("Generate text based on a given prompt.")
st.write("Ponkrit st124960")

prompt = st.text_input("Enter a prompt:", value="Harry Potter is")
temperature = st.slider("Select temperature:", min_value=0.5, max_value=1.5, step=0.1, value=1.0)
max_seq_len = st.slider("Maximum sequence length:", min_value=10, max_value=100, step=10, value=30)

if st.button("Generate"):
    if prompt.strip():
        output = generate_text(prompt, max_seq_len, temperature)
        st.write("**Generated Text:**")
        st.write(output)
    else:
        st.write("Please enter a valid prompt.")
