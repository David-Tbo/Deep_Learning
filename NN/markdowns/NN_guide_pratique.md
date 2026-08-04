## **1. Fondamentaux des Réseaux de Neurones**

### **Architecture**
- **Couches** :
  - **Couche d'entrée** : Reçoit les données brutes (ex: `X = [0.05, 0.10]`).
  - **Couche cachée** : Applique une transformation non-linéaire (ex: ReLU) aux entrées pondérées.
  - **Couche de sortie** : Produit la prédiction finale (ex: Sigmoid pour des valeurs entre 0 et 1).
- **Exemple concret** :
  - 2 entrées → 2 neurones cachés (ReLU) → 2 sorties (Sigmoid).

### **Initialisation**
- **Poids** : Initialisés aléatoirement avec `np.random.randn()`.
- **Biais** : Initialisés à zéro (`np.zeros()`).
- **Important** : Les dimensions des matrices doivent être **compatibles** pour les opérations matricielles (ex: `W_in` de forme `(n_inputs, layer_size)`).

---

---

## **2. Passage Avant (Forward Pass)**
### **Étapes**
1. **Calcul des entrées nettes** :
   - Couche cachée : `net_h = X @ W_in + b_in`
   - Couche de sortie : `net_o = out_h @ W_out + b_out`
2. **Application des activations** :
   - ReLU pour la couche cachée : `out_h = relu(net_h)`
   - Sigmoid pour la sortie : `out_o = sigmoid(net_o)`

### **Fonctions d'activation**
- **ReLU** : `max(0, x)` → Dérivée : `1 si x > 0, sinon 0`.
- **Sigmoid** : `1 / (1 + exp(-x))` → Dérivée : `sigmoid(x) * (1 - sigmoid(x))`.

---

---

## **3. Rétropropagation (Backpropagation)**
### **Principe**
- **Calcul des gradients** : Propager l'erreur de la sortie vers l'entrée en utilisant la **règle de la chaîne**.
- **Mise à jour des poids** : `W -= lr * gradient` (Descente de gradient stochastique, SGD).

### **Formules Clés**
- **Gradient de la perte par rapport à la sortie** :
  `dloss_dout_o = -2 * (y_true - out_o)` (pour MSE).
- **Gradient pour les poids de sortie** :
  `dloss_dW_out = out_h.T @ (dloss_dout_o * sigmoid_derivative(out_o))`.
- **Gradient pour les biais de sortie** :
  `dloss_db_out = np.sum(dloss_dout_o * sigmoid_derivative(out_o), axis=0)`.
  → **Pourquoi `np.sum` ?** : Le biais est un scalaire partagé par tous les neurones de la couche. On somme donc les contributions de chaque neurone.

### **Exemple Matriciel**
Pour une couche cachée avec :
- `dloss_dout_h = [[0.1, 0.2], [0.3, 0.4]]` (gradient de la perte par rapport aux sorties cachées),
- `dout_h_dnet_h = [0.5, 0.6]` (dérivée de ReLU),
alors :
```python
dloss_db_in = np.sum(dloss_dout_h * dout_h_dnet_h, axis=0)
# = [0.1*0.5 + 0.3*0.5, 0.2*0.6 + 0.4*0.6] = [0.2, 0.36]
```

---

---

## **4. Problèmes Courants et Solutions**
### **Erreurs de Dimensions**
- **Problème** : `ValueError: shapes (1,2) and (1,2) not aligned`.
- **Cause** : Incompatibilité des dimensions pour `np.dot()`.
- **Solution** :
  - Vérifier que `X` est de forme `(batch_size, n_inputs)`.
  - Exemple : `X = np.array([[0.05, 0.10]])` (1 échantillon, 2 entrées).

### **Perte Élevée**
- **Causes possibles** :
  1. **Taux d'apprentissage (`lr`) trop grand** → Divergence.
  2. **Initialisation des poids** trop grande → Saturation des activations.
  3. **Erreur dans les gradients** → Vérifier les formules de rétropropagation.
- **Solution** :
  - Réduire `lr` (ex: `0.01` au lieu de `0.1`).
  - Utiliser des opérations matricielles pour éviter les erreurs manuelles.

---

---

## **5. Outils NumPy Utiles**
### **Fonctions Clés**
- **`np.clip(array, a_min, a_max)`** :
  - Limite les valeurs de `array` entre `a_min` et `a_max`.
  - Exemple : `np.clip(x, 1e-16, 1-1e-16)` évite les valeurs exactes 0 ou 1 (pour la stabilité numérique).
  - **Pourquoi `1.0` dans votre exemple ?**
    `1 - np.finfo(float).eps` est si proche de 1 que NumPy l'affiche comme `1.0` (précision flottante), mais la valeur exacte est `0.9999999999999999`.

- **`np.finfo(float).eps`** :
  - Plus petit nombre flottant tel que `1.0 + eps != 1.0` (≈ `2.22e-16`).
  - Utilisé pour éviter les divisions par zéro ou les logarithmes de zéro.

- **`np.dot()` vs `@`** :
  - Les deux calculent le produit matriciel, mais `@` est plus lisible (Python ≥ 3.5).

---

---

## **6. Bonnes Pratiques**
1. **Reproductibilité** :
   - Toujours fixer les graines (`np.random.seed(0)`, `torch.manual_seed(0)`).
2. **Débogage** :
   - Afficher les formes des matrices (`print(W_in.shape)`) pour vérifier la compatibilité.
   - Vérifier les gradients avec des valeurs simples (ex: `X = [1, 1]`).
3. **Optimisation** :
   - Préférer les **opérations matricielles** (vectorisées) aux boucles pour la performance.
   - Exemple : `np.dot(X.T, gradient)` au lieu de boucles sur les neurones.

---

---

## **7. Exemple Complet Corrigé**
```python
import numpy as np

# Initialisation
np.random.seed(0)
X = np.array([[0.05, 0.10]])  # 1 échantillon, 2 entrées
target = np.array([[0.01, 0.99]])  # 1 échantillon, 2 sorties

# Architecture : 2 entrées → 2 neurones cachés (ReLU) → 2 sorties (Sigmoid)
W_in = np.random.randn(2, 2)  # (n_inputs, layer_size)
b_in = np.zeros(2)
W_out = np.random.randn(2, 2)  # (layer_size, out_size)
b_out = np.zeros(2)

# Forward pass
net_h = X @ W_in + b_in
out_h = np.maximum(0, net_h)  # ReLU
net_o = out_h @ W_out + b_out
out_o = 1 / (1 + np.exp(-net_o))  # Sigmoid

# Backpropagation (MSE)
dloss_dout_o = -2 * (target - out_o)
dout_o_dnet_o = out_o * (1 - out_o)
dloss_dW_out = out_h.T @ (dloss_dout_o * dout_o_dnet_o)
dloss_db_out = np.sum(dloss_dout_o * dout_o_dnet_o, axis=0)

dloss_dout_h = (dloss_dout_o * dout_o_dnet_o) @ W_out.T
dout_h_dnet_h = (net_h > 0).astype(float)  # Dérivée ReLU
dloss_dW_in = X.T @ (dloss_dout_h * dout_h_dnet_h)
dloss_db_in = np.sum(dloss_dout_h * dout_h_dnet_h, axis=0)

# Mise à jour des poids
lr = 0.01
W_out -= lr * dloss_dW_out
b_out -= lr * dloss_db_out
W_in -= lr * dloss_dW_in
b_in -= lr * dloss_db_in
```

---

---
## **Résumé Final**
| Concept               | À Retenir                                                                 |
|-----------------------|---------------------------------------------------------------------------|
| **Architecture**      | Couches : Entrée → Cachée (ReLU) → Sortie (Sigmoid).                     |
| **Forward Pass**      | `net = X @ W + b`, puis appliquer l'activation.                          |
| **Backpropagation**   | Calculer les gradients avec la règle de la chaîne, puis `W -= lr * dW`. |
| **Dimensions**        | Vérifier `shape` pour éviter les erreurs de produit matriciel.         |
| **Outils NumPy**      | `np.clip`, `np.dot`, `np.sum(axis=0)` pour les biais.                     |
| **Débogage**          | Afficher les formes et les valeurs intermédiaires.                       |

---
**Prochaines étapes** :
- Implémenter une **boucle d'entraînement** avec plusieurs époques.
- Ajouter un **suivi de la perte** pour vérifier la convergence.
- Tester avec des **données plus complexes** (ex: XOR).