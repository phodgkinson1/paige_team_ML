import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import re
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the LSTM Model
class LSTMTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150, hidden_dim=512, output_dim=None):
        super(LSTMTextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Account for bidirectionality
    def attention(self, lstm_out):
        attention_weights = torch.tanh(self.attention_fc(lstm_out))
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_output = torch.sum(attention_weights * lstm_out, dim=1)
        return weighted_output
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        attention_out = self.attention(x)   
        x = self.fc(attention_out) 
        return x

# Load the pre-trained model
model_path = "lstm_word_generation_model.pth"
model = LSTMTextGenerationModel(len(tokenizer.word_index) + 1, embedding_dim=150, hidden_dim=512, output_dim=len(tokenizer.word_index) + 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded and ready for text generation.")

# Define the text generation function with temperature scaling and top-k sampling
def generate_text(seed_text, next_words=100, temperature=0.7, top_k=10):
    # Filter non-essential characters
    seed_text = re.sub(r'[^A-Za-z0-9.,\'\s-]', '', seed_text)
    generated_text = seed_text
    model.eval()
    
    with torch.no_grad():
        for _ in range(next_words):
            # Tokenize and pad the input sequence to the required length
            tokenized_input = tokenizer.texts_to_sequences([generated_text.split()])
            tokenized_input = pad_sequences([tokenized_input[0]], maxlen=40, padding='pre')
            tokenized_input = torch.tensor(tokenized_input, dtype=torch.long).to(device)
            
            output = model(tokenized_input).squeeze()
            output = output / temperature
            # Apply top-k sampling for diverse generation
            top_k_prob, top_k_indices = torch.topk(F.softmax(output, dim=-1), top_k)
            next_word_idx = top_k_indices[torch.multinomial(top_k_prob, 1).item()]
            next_word = tokenizer.index_word.get(next_word_idx.item(), '')
            if not next_word:  # If no next word is found, end generation
                break
            generated_text += ' ' + next_word
    return generated_text

# Prompt for seed text and generate output
seed_text = input("Enter seed prompt: ")
generated_output = generate_text(seed_text, next_words=200, temperature=0.7)
print("\nGenerated Text:")
print(generated_output)
