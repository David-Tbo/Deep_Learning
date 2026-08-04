Voici les **codes PyTorch essentiels** à retenir organisés par cas d'usage.

---

## **1. Perceptron Simple (1 couche, 1 neurone)**
**Cas d'usage** : Réseau à une seule couche avec activation sigmoïde (classification binaire ou régression simple).

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Configuration
torch.manual_seed(0)
X = torch.tensor([[0.5, 0.3, 0.8]], dtype=torch.float32)  # 1 échantillon, 3 features
target = torch.tensor([[0.9]], dtype=torch.float32)      # Target

# Modèle
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)  # 1 neurone de sortie
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = Perceptron(input_size=3)
criterion = nn.MSELoss()  # Loss MSE
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Entraînement
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Prédiction
with torch.no_grad():
    print(f"Output: {model(X).item():.4f}")
    print(f"Final Loss: {criterion(model(X), target).item():.4f}")
```

**Points clés** :
- `nn.Linear` pour la transformation linéaire.
- `nn.Sigmoid` pour l'activation.
- `MSELoss` pour la régression, `BCELoss` pour la classification binaire.

---

---

## **2. Réseau à 1 Couche Cachée (ReLU + Sigmoïde)**
**Cas d'usage** : Réseau avec une couche cachée (ReLU) et une sortie sigmoïde.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Données
torch.manual_seed(0)
X = torch.tensor([[0.1, 0.5]], dtype=torch.float32)  # 1 échantillon, 2 features
target = torch.tensor([[0.2]], dtype=torch.float32)

# Modèle
class NNSimple(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Couche cachée
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)          # Couche de sortie
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = NNSimple(input_size=2, hidden_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Entraînement
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Résultats
with torch.no_grad():
    print(f"Output: {model(X).item():.4f}")
    print(f"Final Loss: {criterion(model(X), target).item():.4f}")
```

**Points clés** :
- Architecture : `Input → ReLU → Sigmoid`.
- `nn.ReLU` pour la couche cachée (meilleure performance que Sigmoid pour les couches cachées).
- Utilisation de `zero_grad()` pour réinitialiser les gradients à chaque itération.

---

---

## **3. Version Générique (Batch Training)**
**Cas d'usage** : Entraînement sur un batch de données (plus réaliste).

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Données (batch de 4 échantillons)
torch.manual_seed(0)
X = torch.tensor([
    [0.1, 0.5],
    [0.2, 0.4],
    [0.3, 0.6],
    [0.4, 0.7]
], dtype=torch.float32)
target = torch.tensor([[0.2], [0.3], [0.4], [0.5]], dtype=torch.float32)

# Modèle
class NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

model = NN(input_size=2, hidden_size=2)  # 2 neurones cachés
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Entraînement
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Prédiction
with torch.no_grad():
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input {X[i].tolist()} → Output: {model(X[i:i+1]).item():.4f}")
```

**Points clés** :
- Gestion de **batches** (plusieurs échantillons à la fois).
- Affichage de la perte pendant l'entraînement.
- Prédiction individuelle pour chaque échantillon.

---

---

## **4. Version avec Validation et Early Stopping**
**Cas d'usage** : Ajout d'une validation et arrêt précoce si la perte ne diminue plus.

```python
import torch
import numpy as np

# Données
torch.manual_seed(0)
X_train = torch.rand(100, 2)  # 100 échantillons d'entraînement
y_train = torch.rand(100, 1)
X_val = torch.rand(20, 2)    # 20 échantillons de validation
y_val = torch.rand(20, 1)

# Modèle
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = NN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Early Stopping
best_loss = float('inf')
patience = 5
no_improve = 0

for epoch in range(1000):
    # Entraînement
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

    # Early Stopping
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

**Points clés** :
- **Validation** : Évaluation sur un jeu de données séparé.
- **Early Stopping** : Arrêt si la perte de validation n'améliore pas pendant `patience` époques.
- Bonnes pratiques pour éviter le surapprentissage.

---

---

## **Résumé des Bonnes Pratiques**
1. **Initialisation** :
   - Toujours utiliser `torch.manual_seed(0)` pour la reproductibilité.
   - Convertir les données en `torch.tensor` avec `dtype=torch.float32`.

2. **Modélisation** :
   - Utiliser `nn.Module` pour définir des architectures personnalisées.
   - Préférer `ReLU` pour les couches cachées et `Sigmoid`/`Softmax` pour la sortie.

3. **Entraînement** :
   - Toujours appeler `zero_grad()` avant `backward()`.
   - Utiliser `with torch.no_grad()` pour les prédictions (désactive le calcul des gradients).

4. **Optimisation** :
   - `SGD` pour un contrôle fin du taux d'apprentissage.
   - `Adam` pour une optimisation plus robuste (remplace `SGD` par `optim.Adam`).

5. **Évaluation** :
   - Utiliser `MSELoss` pour la régression, `BCELoss` (avec Sigmoid) ou `CrossEntropyLoss` (avec Softmax) pour la classification.

---
---
### **Code à Retenir Absolument**
Si tu ne devais retenir qu'un seul code, ce serait cette version **générique, modulaire et prête à l'emploi** :

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Préparation des données
torch.manual_seed(0)
X = torch.rand(100, 3)  # 100 échantillons, 3 features
y = torch.rand(100, 1)  # Target

# 2. Définition du modèle
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 3. Initialisation
model = SimpleNN(input_size=3, hidden_size=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam est souvent plus efficace que SGD

# 4. Entraînement
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. Prédiction
with torch.no_grad():
    print(f"Output exemple: {model(X[:1]).item():.4f}")
```

**Pourquoi ce code ?**
- **Modulaire** : Facile à adapter (changer les tailles des couches, l'optimiseur, etc.).
- **Robuste** : Utilise `Adam` (meilleur que SGD dans la plupart des cas).
- **Complet** : Couvre tout le pipeline (données → modèle → entraînement → prédiction).