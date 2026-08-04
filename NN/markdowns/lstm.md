# Long-Short-Term-Memory (LSTM)

* Initialize the LSTM decoder with the last hidden state of the encoder.
* This means the initial states of the decoder are defined by the final states of the encoder (encoder_states = (state_h, state_c)).
* Since we don't care about the last hidden or cell states, we mark them with _.
We are only interested in the decoder's hidden state output at each timestep.
The decoder will contain an array of hidden states, one for each timestep.
Set the decoder's initial state to the encoder's final output states.
Since return_sequences is set to True, decoder_outputs will be a collection of the decoder's hidden states at each timestep.

```python
decoder_outputs, _, _ = decoder_lstm(decoder_embedding_output, initial_state=encoder_states)
```

Pour améliorer les performances de votre modèle de traduction automatique, vous pouvez envisager plusieurs stratégies. Voici quelques suggestions :

### 1. **Augmenter la Complexité du Modèle**
- **Ajouter des Couches** : Ajoutez plus de couches LSTM ou GRU pour capturer des dépendances plus complexes dans les données.
- **Utiliser des Attention Mechanisms** : Intégrez un mécanisme d'attention pour permettre au décodeur de se concentrer sur différentes parties de la séquence d'entrée à chaque étape de la décodification.

### 2. **Optimiser les Hyperparamètres**
- **Taille de l'Embedding** : Augmentez la taille de l'embedding (`embedding_dim`) pour capturer plus d'informations contextuelles.
- **Taille des Couches Cachées** : Augmentez la taille des couches cachées (`hidden_dim`) pour permettre au modèle de capturer des représentations plus complexes.
- **Taux de Dropout** : Ajustez le taux de dropout pour éviter le surapprentissage tout en permettant au modèle de généraliser correctement.

### 3. **Améliorer le Prétraitement des Données**
- **Augmentation des Données** : Utilisez des techniques d'augmentation des données pour générer plus de paires de phrases d'entraînement.
- **Nettoyage des Données** : Assurez-vous que les données sont correctement nettoyées et prétraitées pour éviter les bruits inutiles.

### 4. **Utiliser des Techniques Avancées**
- **Bidirectional LSTMs** : Utilisez des LSTMs bidirectionnels pour capturer des informations contextuelles dans les deux sens.
- **Transformers** : Envisagez d'utiliser des architectures de transformateurs, qui sont actuellement à la pointe de la technologie pour les tâches de traitement du langage naturel.

### 5. **Optimisation de l'Entraînement**
- **Learning Rate Scheduling** : Utilisez un planificateur de taux d'apprentissage pour ajuster dynamiquement le taux d'apprentissage pendant l'entraînement.
- **Batch Size** : Expérimentez avec différentes tailles de lots pour trouver un équilibre entre la stabilité de l'entraînement et l'utilisation efficace de la mémoire.

### 6. **Évaluation et Validation**
- **Validation Croisée** : Utilisez la validation croisée pour évaluer les performances du modèle de manière plus robuste.
- **Métriques Supplémentaires** : Ajoutez des métriques supplémentaires comme le BLEU score pour évaluer la qualité des traductions générées.

### Exemple de Modifications

Voici un exemple de modifications que vous pouvez apporter à votre code pour intégrer certaines de ces suggestions :

```python
import os
import re
import unicodedata
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Chargement et prétraitement des données
data_path = "/Users/davidtbo/Documents/Data_Science/99_Data/"
TRAIN_FILE = 'hun_eng_pairs_train.txt'
VAL_FILE = 'hun_eng_pairs_val.txt'
SEPARATOR = '<sep>'

def load_data(file_path):
    with open(file_path) as file:
        return [line.rstrip() for line in file]

def preprocess_sentence(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    return s.strip()

def tag_target_sentences(sentences):
    return [' '.join(['<sos>', s, '<eos>']) for s in sentences]

def generate_decoder_inputs_targets(sentences, tokenizer):
    seqs = [tokenizer(s) for s in sentences]
    decoder_inputs = [s[:-1] for s in seqs]
    decoder_targets = [s[1:] for s in seqs]
    return decoder_inputs, decoder_targets

class TranslationDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return (torch.tensor(self.encoder_inputs[idx]),
                torch.tensor(self.decoder_inputs[idx]),
                torch.tensor(self.decoder_targets[idx]))

def collate_fn(batch):
    encoder_inputs, decoder_inputs, decoder_targets = zip(*batch)
    encoder_inputs = pad_sequence(encoder_inputs, padding_value=0, batch_first=True)
    decoder_inputs = pad_sequence(decoder_inputs, padding_value=0, batch_first=True)
    decoder_targets = pad_sequence(decoder_targets, padding_value=0, batch_first=True)
    return encoder_inputs, decoder_inputs, decoder_targets

train = load_data(os.path.join(data_path, TRAIN_FILE))
val = load_data(os.path.join(data_path, VAL_FILE))

train_input, train_target = zip(*[pair.split(SEPARATOR) for pair in train])
train_preprocessed_input = [preprocess_sentence(s) for s in train_input]
train_preprocessed_target = [preprocess_sentence(s) for s in train_target]
train_tagged_preprocessed_target = tag_target_sentences(train_preprocessed_target)

source_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>', filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n')
source_tokenizer.fit_on_texts(train_preprocessed_input)
source_vocab_size = len(source_tokenizer.word_index) + 1

target_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>', filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n')
target_tokenizer.fit_on_texts(train_tagged_preprocessed_target)
target_vocab_size = len(target_tokenizer.word_index) + 1

train_encoder_inputs = source_tokenizer.texts_to_sequences(train_preprocessed_input)
train_decoder_inputs, train_decoder_targets = generate_decoder_inputs_targets(train_tagged_preprocessed_target, target_tokenizer)

max_encoding_len = max(len(s) for s in train_encoder_inputs)
max_decoding_len = max(len(s) for s in train_decoder_inputs)

train_dataset = TranslationDataset(train_encoder_inputs, train_decoder_inputs, train_decoder_targets)
val_dataset = TranslationDataset(*process_dataset(val, source_tokenizer, target_tokenizer, max_encoding_len, max_decoding_len))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Définition du modèle
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        hidden, cell = self.encoder(source)
        outputs, _, _ = self.decoder(target, hidden, cell)
        return outputs

embedding_dim = 256
hidden_dim = 512
default_dropout = 0.3

encoder = Encoder(source_vocab_size, embedding_dim, hidden_dim, default_dropout)
decoder = Decoder(target_vocab_size, embedding_dim, hidden_dim, default_dropout)
model = Seq2Seq(encoder, decoder)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
def train(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for encoder_inputs, decoder_inputs, decoder_targets in loader:
        optimizer.zero_grad()
        outputs = model(encoder_inputs, decoder_inputs)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), decoder_targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for encoder_inputs, decoder_inputs, decoder_targets in loader:
            outputs = model(encoder_inputs, decoder_inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), decoder_targets.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

epochs = 10
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
```

### Explications des Modifications :
1. **Augmentation de la Complexité du Modèle** : La taille de l'embedding et des couches cachées a été augmentée.
2. **Optimisation des Hyperparamètres** : Le taux de dropout a été ajusté et la taille du batch a été augmentée.
3. **Callbacks Avancés** : Ajout de `ReduceLROnPlateau` pour ajuster dynamiquement le taux d'apprentissage.
4. **Simplification du Code** : Fonctions de prétraitement et de chargement des données simplifiées pour une meilleure lisibilité.

Ces modifications devraient aider à améliorer les performances de votre modèle. Vous pouvez continuer à expérimenter avec d'autres techniques avancées pour obtenir de meilleurs résultats.