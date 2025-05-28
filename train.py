import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import re
import pickle
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load a subset of WikiText-103 dataset
print("Loading WikiText-103 dataset...")
# dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:15%]")  # Reduced dataset size
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

shuffled_dataset = dataset.shuffle(seed=9721) 

sampled_dataset = shuffled_dataset.select(range(int(0.33 * len(shuffled_dataset))))

print(f"Original dataset size: {len(dataset)}")
print(f"Sampled dataset size: {len(sampled_dataset)}, {100*len(sampled_dataset)/len(dataset):.1f}% dataset")
text_samples = sampled_dataset['text']
text = " ".join(text_samples)

# Filter out unwanted symbols
print("Filtering text...")
filtered_text = re.sub(r'[^A-Za-z0-9.,\'\s-]', '', text)
print(f"Filtered text length: {len(filtered_text)} characters")

# Tokenize text
print("Tokenizing text...")
tokenizer = Tokenizer()  # Word-level tokenizer
tokenizer.fit_on_texts([filtered_text])
total_words = len(tokenizer.word_index) + 1
print(f"Total unique words: {total_words}")

# Convert text to a sequence of integers
print("Converting text to integer sequence...")
input_sequence = tokenizer.texts_to_sequences([filtered_text])[0]

# Define sequence length and steps per epoch
sequence_length = 40  # Reduced sequence length
steps_per_epoch = 1
sequences = []
next_words = []

# Generate sequences for training
print("Generating sequences and next words for training...")
for i in range(0, len(input_sequence) - sequence_length, steps_per_epoch):
    sequences.append(input_sequence[i:i + sequence_length])
    next_words.append(input_sequence[i + sequence_length])
sequences = pad_sequences(sequences, maxlen=sequence_length, padding='pre')

# Convert sequences and targets to PyTorch tensors and move to device
X = torch.tensor(sequences, dtype=torch.long).to(device)
y = torch.tensor(next_words, dtype=torch.long).to(device)

# Define PyTorch Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)  # Reduced batch size

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

def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:.0f}m, {seconds:.0f}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:.0f}h, {minutes:.0f}m, {seconds:.0f}s"

# Instantiate the model, loss function, and optimizer
embedding_dim = 150
hidden_dim = 512
model = LSTMTextGenerationModel(total_words, embedding_dim, hidden_dim, total_words).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)  # Updated scheduler parameters

# Training loop with gradient accumulation
print("Starting training...")
num_epochs = 10  # Reduced epochs for initial testing
accumulation_steps = 2  # Number of mini-batches to accumulate gradients

loss_history = []
perplexity_history = []
train_start_time = time.time() 
for epoch in range(num_epochs):
    total_loss = 0
    total_tokens = 0
    model.train()
    epoch_start_time = time.time() 

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        batch_start_time = time.time()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss / accumulation_steps  # Normalize loss by accumulation steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # Apply gradient clipping
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Scale back up for tracking
        total_tokens += targets.numel()

        if (batch_idx + 1) % 100 == 0:
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - train_start_time
            remaining_batches = ((10-epoch)*len(dataloader)) - (batch_idx + 1)
            estimated_time_left = batch_time * remaining_batches

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {loss.item() * accumulation_steps:.4f}, "
                  f"Elapsed Time: {format_seconds(elapsed_time)}, Estimated Time Left: {format_seconds(estimated_time_left)}")
        
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    perplexity = math.exp(total_loss / total_tokens)
    perplexity_history.append(perplexity)
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{num_epochs}] completed with average loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f} in {format_seconds(epoch_duration)}")

elapsed_time = time.time() - train_start_time
print(f"Training complete in {format_seconds(elapsed_time)}.")
model_path = "lstm_word_generation_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save tokenizer for use in generation
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved to tokenizer.pkl")

# Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
# Plot the Perplexity history
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), perplexity_history, marker='o', label='Training Perplexity')
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Perplexity Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
