import torch
import torch.nn as nn
import streamlit as st
import spacy
from pythainlp.tokenize import word_tokenize
import os

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp_en = load_spacy()

def tokenize_en(text):
    return [token.text for token in nlp_en(text.lower().strip())]

def tokenize_th(text):
    return word_tokenize(text.lower().strip())

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        hidden = (hidden[0] + hidden[1]) / 2
        cell = (cell[0] + cell[1]) / 2

        outputs = self.linear(outputs)

        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = attention
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return prediction, hidden.squeeze(0), cell.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, trg=None, max_len=50):
        encoder_outputs, hidden, cell = self.encoder(src, src_len)

        if trg is None:  # Inference Mode
            input_ = torch.tensor([vocab_th['<sos>']], dtype=torch.long, device=src.device)
            translated_sentence = []

            for _ in range(max_len):  # Generate sentence step by step
                output, hidden, cell = self.decoder(input_, hidden, cell, encoder_outputs)
                top1 = output.argmax(1).item()
                if top1 == vocab_th['<eos>']:
                    break
                translated_sentence.append(top1)
                input_ = torch.tensor([top1], dtype=torch.long, device=src.device)

            return translated_sentence  # Return list of indices

        else:  # Training Mode
            input_ = trg[:, 0]
            outputs = []

            for t in range(1, trg.shape[1]):
                output, hidden, cell = self.decoder(input_, hidden, cell, encoder_outputs)
                outputs.append(output)
                input_ = output.argmax(1)

            return torch.stack(outputs, dim=1)  # Return full sequence
class MultiplicativeAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.W = nn.Linear(dec_hid_dim, enc_hid_dim, bias=False)

    def forward(self, s, h):
        s = self.W(s).unsqueeze(2)
        attn_scores = torch.bmm(h, s).squeeze(2)
        return torch.softmax(attn_scores, dim=1)

# Load Model and Vocabulary (cached)
@st.cache_resource
def load_model_and_vocabs():
    vocab_en_path = "vocab_en.pth"
    vocab_th_path = "vocab_th.pth"

    vocab_en = torch.load(vocab_en_path)
    vocab_th = torch.load(vocab_th_path)

    # Ensure vocab has the required tokens
    required_tokens = ["<sos>", "<eos>", "<unk>", "<pad>"]
    for token in required_tokens:
        if token not in vocab_en or token not in vocab_th:
            raise KeyError(f"Missing token {token} in vocab files. Ensure vocab was built correctly.")

    emb_dim = 256
    hid_dim = 512
    dropout = 0.5

    encoder = Encoder(len(vocab_en), emb_dim, hid_dim, dropout)
    attention = MultiplicativeAttention(hid_dim, hid_dim)
    decoder = Decoder(len(vocab_th), emb_dim, hid_dim, hid_dim, dropout, attention)
    model = Seq2Seq(encoder, decoder)

    # Load model weights
    state_dict = torch.load("best_model_multiplicative.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, vocab_en, vocab_th

model, vocab_en, vocab_th = load_model_and_vocabs()


def text_pipeline_en(text):
    tokens = tokenize_en(text)
    if vocab_en is None:
        raise ValueError("Error: vocab_en is None. Ensure vocab_en.pth is loaded correctly.")

    indices = [vocab_en['<sos>']] + [vocab_en[token] if token in vocab_en.get_stoi() else vocab_en['<unk>'] for token in tokens] + [vocab_en['<eos>']]

    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def text_pipeline_th(text):
    tokens = tokenize_th(text)
    if vocab_th is None:
        raise ValueError("Error: vocab_th is None. Ensure vocab_th.pth is loaded correctly.")

    indices = [vocab_th['<sos>']] + [vocab_th[token] if token in vocab_th.get_stoi() else vocab_th['<unk>'] for token in tokens] + [vocab_th['<eos>']]

    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def translate(sentence):
    model.eval()
    src = text_pipeline_en(sentence)
    src_len = torch.tensor([src.shape[1]], dtype=torch.int64)

    with torch.no_grad():
        translated_indices = []
        encoder_outputs, hidden, cell = model.encoder(src, src_len)
        input_ = torch.tensor([vocab_th['<sos>']], dtype=torch.long)

        for _ in range(50):  # Max translation length
            output, hidden, cell = model.decoder(input_, hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()

            if top1 == vocab_th['<eos>']:  # Stop when EOS token is reached
                break

            translated_indices.append(top1)
            input_ = torch.tensor([top1], dtype=torch.long)

    return " ".join([vocab_th.lookup_token(idx) for idx in translated_indices])


# Streamlit App
st.title("English-to-Thai Machine Translation")
st.write("Ponkrit Kaewsawee st124960")
input_text = st.text_input("Enter English Sentence:")

if st.button("Translate"):
    if input_text:
        translation = translate(input_text)
        st.write("**Translation:**", translation)
    else:
        st.write("Please enter a sentence to translate.")