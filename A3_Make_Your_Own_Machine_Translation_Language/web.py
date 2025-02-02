import torch
import streamlit as st
import spacy
from pythainlp.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence

# ✅ Load Tokenizer
nlp_en = spacy.load("en_core_web_sm")

def tokenize_en(text):
    return [token.text for token in nlp_en(text.lower().strip())]

def tokenize_th(text):
    return word_tokenize(text.lower().strip())

# ✅ Load Model and Vocabulary
@st.cache_resource()
def load_model():
    encoder = torch.load("encoder.pth", map_location=torch.device("cpu"))
    decoder = torch.load("decoder.pth", map_location=torch.device("cpu"))
    vocab_en = torch.load("vocab_en.pth")
    vocab_th = torch.load("vocab_th.pth")
    return encoder, decoder, vocab_en, vocab_th

encoder, decoder, vocab_en, vocab_th = load_model()

def text_pipeline_en(text):
    tokens = tokenize_en(text)
    tensor = torch.tensor([vocab_en['<sos>']] + [vocab_en[token] for token in tokens] + [vocab_en['<eos>']], dtype=torch.long)
    return tensor.unsqueeze(0)

def translate(sentence):
    encoder.eval()
    decoder.eval()
    
    src = text_pipeline_en(sentence)
    src_len = torch.tensor([src.shape[1]], dtype=torch.int64)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = encoder(src, src_len)
        input_ = torch.tensor([vocab_th['<sos>']], dtype=torch.long)
        translated_sentence = []
        
        for _ in range(50):  # Max translation length
            output, hidden, cell = decoder(input_, hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()
            if top1 == vocab_th['<eos>']:
                break
            translated_sentence.append(top1)
            input_ = torch.tensor([top1], dtype=torch.long)
    
    return " ".join([vocab_th.lookup_token(idx) for idx in translated_sentence])

# ✅ Web App Interface
st.title("English-to-Thai Machine Translation")
input_text = st.text_input("Enter English Sentence:")
if st.button("Translate"):
    translation = translate(input_text)
    st.write("**Translation:**", translation)
