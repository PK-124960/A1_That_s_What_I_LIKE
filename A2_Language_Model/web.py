import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from your_model_code import LSTMLanguageModel  # Import your trained model

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your trained model (make sure to save it as 'best-val-lstm_lm.pt' after training)
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load('best-val-lstm_lm.pt'))
model.eval()

# Define text generation function
def generate_text(prompt, model, vocab, seq_len=50, temperature=1.0):
    # Convert the prompt to token indices
    tokens = [vocab[token] for token in prompt.split()]

    # Create tensor input for the model
    input_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    hidden = model.init_hidden(1, device)

    output = prompt
    for _ in range(seq_len):
        with torch.no_grad():
            prediction, hidden = model(input_tensor, hidden)
        
        # Get the last word prediction
        last_word_logits = prediction[0, -1, :]
        probabilities = torch.softmax(last_word_logits / temperature, dim=-1)
        next_token_idx = torch.multinomial(probabilities, 1).item()
        
        # Convert the token index back to a word
        next_token = vocab.lookup_token(next_token_idx)
        output += ' ' + next_token

        # Update the input tensor for the next prediction
        input_tensor = torch.cat([input_tensor, torch.LongTensor([next_token_idx]).unsqueeze(0).to(device)], dim=1)

    return output

# Streamlit app layout
st.title("Harry Potter Text Generator")
st.text_area("Enter a prompt:", "Harry Potter is", height=100)
prompt = st.text_input("Prompt:", "Harry Potter is")

if st.button("Generate Text"):
    if prompt:
        generated_text = generate_text(prompt, model, vocab)
        st.write(generated_text)
    else:
        st.write("Please enter a prompt to generate text.")

