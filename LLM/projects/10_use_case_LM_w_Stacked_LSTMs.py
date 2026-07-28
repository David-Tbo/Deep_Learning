"""
Character-Level Language Modeling with Stacked LSTMs (PyTorch Implementation)
=============================================================================
Description:
    This script implements a character-level auto-regressive language model using a 
    stacked Bidirectional/Unidirectional LSTM architecture in PyTorch. The model is trained 
    on Sun Tzu's "The Art of War" to generate text character-by-character based on a seed prompt.

Key Components & Pipeline:
    1. Corpus Acquisition: Downloads raw text corpus dynamically via `requests`.
    2. Character Tokenization & Vocabulary: Builds a mapping between characters and 
       integer indices, replacing Keras's character-level Tokenizer.
    3. Sliding Window Dataset Creation: Converts the text sequence into sliding windows 
       of length `input_timesteps + 1` to create paired Input (x) and Target (y) sequences 
       shifted by one character (Teacher Forcing).
    4. One-Hot Vectorization: Maps input character IDs to One-Hot encoded vectors.
    5. Stacked LSTM Architecture:
       - Multi-layer LSTM with recurrent dropout equivalent (`dropout` parameter in PyTorch LSTM).
       - Linear Output Layer projecting hidden state outputs to raw character logits.
    6. Text Generation Engine: Uses temperature-scaled Softmax sampling (`torch.multinomial`) 
       to generate autoregressive text continuations from arbitrary seed prompts.

Framework: PyTorch (Pure Python/PyTorch execution, no Keras/TensorFlow dependencies).
"""

import requests
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Select compute hardware (GPU acceleration if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using computational device: {device}")


# =============================================================================
# 1. CORPUS ACQUISITION & CHAR-LEVEL TOKENIZATION
# =============================================================================

# Download Sun Tzu's "The Art of War" corpus
url = "https://raw.githubusercontent.com/futuremojo/nlp-demystified/main/datasets/art_of_war.txt"
art_of_war = requests.get(url).text
print(f"Corpus Loaded. Total character length: {len(art_of_war)}")


class CharTokenizer:
    """
    Handles character-level tokenization and bidirectional mapping 
    between raw characters and unique integer IDs.
    """
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def fit_on_text(self, text: str):
        """Constructs vocabulary lookup tables based on unique characters."""
        unique_chars = sorted(list(set(text)))
        for idx, char in enumerate(unique_chars):
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def text_to_sequences(self, text: str) -> list:
        """Converts string of characters to list of integer indices."""
        return [self.char2idx[char] for char in text if char in self.char2idx]

    def sequences_to_text(self, indices: list) -> str:
        """Converts list of integer indices back to string."""
        return "".join([self.idx2char[idx] for idx in indices])

    def __len__(self):
        return len(self.char2idx)


# Initialize and fit tokenizer on the text corpus
tokenizer = CharTokenizer()
tokenizer.fit_on_text(art_of_war)

num_tokens = len(tokenizer)
print(f"Unique character vocabulary size: {num_tokens}")

# Convert complete corpus string into integer indices
encoded_corpus = tokenizer.text_to_sequences(art_of_war)


# =============================================================================
# 2. PYTORCH DATASET & SLIDING WINDOW DATALOADER
# =============================================================================

class CharWindowDataset(Dataset):
    """
    PyTorch Dataset implementing sliding window extraction for Language Modeling.
    
    Given a sequence of length N, extracts overlapping sub-sequences of size 
    `input_timesteps + 1`. 
    - Input sequence (x): tokens from index 0 to input_timesteps - 1
    - Target sequence (y): tokens from index 1 to input_timesteps (shifted by +1)
    """
    def __init__(self, encoded_data: list, input_timesteps: int = 100):
        self.encoded_data = encoded_data
        self.input_timesteps = input_timesteps
        self.total_windows = len(encoded_data) - input_timesteps

    def __len__(self):
        return max(0, self.total_windows)

    def __getitem__(self, idx: int):
        # Extract window of length (input_timesteps + 1)
        chunk = self.encoded_data[idx : idx + self.input_timesteps + 1]
        
        # Teacher Forcing alignment: Shifted targets by 1 step
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        # One-Hot encode inputs (Shape: [seq_len, num_tokens])
        x_one_hot = nn.functional.one_hot(x, num_classes=num_tokens).float()
        
        return x_one_hot, y


# Define Sequence & Batch Hyperparameters
INPUT_TIMESTEPS = 100
BATCH_SIZE = 64

# Instantiate Dataset and DataLoader
dataset = CharWindowDataset(encoded_corpus, input_timesteps=INPUT_TIMESTEPS)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"Total training examples created: {len(dataset)}")


# =============================================================================
# 3. STACKED LSTM NEURAL NETWORK ARCHITECTURE
# =============================================================================

class StackedLSTMLanguageModel(nn.Module):
    """
    Stacked Autoregressive LSTM Model for Character Generation.
    
    Data Flow:
    1. Input Tensor: One-Hot encoded characters -> [Batch Size, Seq Length, Num Tokens]
    2. Stacked LSTM: 2 LSTM layers processing sequentially over time.
    3. Linear Projection: Maps hidden dimension to output vocabulary logits -> [Batch Size, Seq Length, Num Tokens]
    """
    def __init__(self, num_tokens: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # Multi-layer stacked LSTM network
        self.lstm = nn.LSTM(
            input_size=num_tokens,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Linear layer mapping hidden states back to vocabulary distribution logits
        self.fc = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x: torch.Tensor, hidden=None) -> tuple:
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Project each timestep output to character class logits
        logits = self.fc(lstm_out)  # Shape: [batch_size, seq_len, num_tokens]
        
        return logits, hidden


# Instantiate Model, Loss Function, and Optimizer
model = StackedLSTMLanguageModel(
    num_tokens=num_tokens, 
    hidden_dim=128, 
    num_layers=2, 
    dropout=0.2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print("\nModel Summary:")
print(model)


# =============================================================================
# 4. TRAINING LOOP ROUTINE
# =============================================================================

def train_model(model, loader, criterion, optimizer, device, epochs=5):
    """Executes model training over the character sequence dataset."""
    model.train()
    print("\nStarting Training...")
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: obtain logits across all timesteps
            logits, _ = model(x_batch)  # Shape: [batch_size, seq_len, num_tokens]
            
            # Reshape for CrossEntropyLoss:
            # Logits -> [(batch_size * seq_len), num_tokens]
            # Targets -> [(batch_size * seq_len)]
            logits_flat = logits.view(-1, num_tokens)
            y_flat = y_batch.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx + 1}/{len(loader)}] | Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(loader)
        print(f"---> Epoch {epoch} Complete | Average Loss: {epoch_loss:.4f}\n")


# Train the model for demonstration (Adjust EPOCHS as needed)
EPOCHS = 3
train_model(model, train_loader, criterion, optimizer, device, epochs=EPOCHS)


# =============================================================================
# 5. AUTOREGRESSIVE TEXT GENERATION WITH TEMPERATURE SAMPLING
# =============================================================================

def generate_text(model, tokenizer, seed_text: str, num_chars: int = 200, temperature: float = 1.0) -> str:
    """
    Generates continuation text autoregressively given a starting seed string.
    
    Uses temperature scaling over Softmax probabilities:
    - Temperature < 1.0: Sharpen probabilities (more conservative/repetitive).
    - Temperature > 1.0: Flatten probabilities (more random/creative).
    """
    model.eval()
    generated_text = seed_text

    with torch.no_grad():
        for _ in range(num_chars):
            # Take the last `INPUT_TIMESTEPS` characters from current text
            input_chars = generated_text[-INPUT_TIMESTEPS:]
            encoded_input = tokenizer.text_to_sequences(input_chars)
            
            # Convert to PyTorch tensor
            input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0) # [1, seq_len]
            
            # One-Hot encode input
            input_one_hot = nn.functional.one_hot(input_tensor, num_classes=num_tokens).float().to(device)
            
            # Forward pass to obtain next-character logits
            logits, _ = model(input_one_hot)
            
            # Extract logits of the very LAST timestep
            last_logits = logits[0, -1, :] # Shape: [num_tokens]
            
            # Apply temperature scaling
            scaled_logits = last_logits / max(temperature, 1e-5)
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Sample next character index according to adjusted probability distribution
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Decode index and append to running text
            next_char = tokenizer.sequences_to_text([next_char_idx])
            generated_text += next_char

    return generated_text


# =============================================================================
# 6. INFERENCE & EXPERIMENTS WITH TEMPERATURE
# =============================================================================

print("\n================ TEXT GENERATION SAMPLES ================")

prompts = [
    ("Banana peels on the battlefield can", 0.2),
    ("It's time to release the Kraken when", 0.5),
    ("Crush your enemies, see them driven before you, and", 0.8),
    ("What is best in life?", 1.2)
]

for seed, temp in prompts:
    print(f"\n--- Seed: '{seed}' | Temperature: {temp} ---")
    output = generate_text(model, tokenizer, seed_text=seed, num_chars=200, temperature=temp)
    print(output)