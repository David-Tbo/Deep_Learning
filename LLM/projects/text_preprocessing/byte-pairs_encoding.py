import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")
sentence = "Unbelievable tokenization works differently."
doc = nlp(sentence)

# 1. On compte les fréquences de chaque mot trouvé par spaCy
word_counts = Counter([t.text for t in doc])

# 2. On prépare le vocabulaire au format BPE : 
# "works" devient "w o r k s </w>"
vocab = {}
for word, freq in word_counts.items():
    # Sépare chaque lettre par un espace et ajoute le marqueur de fin de mot
    bpe_word = " ".join(list(word)) + " </w>"
    vocab[bpe_word] = freq

print("--- Notre vocabulaire au format BPE ---")
print(vocab)
print("\n--------------------------------------")

# 3. Votre fonction get_pair_stats (qui fonctionne maintenant parfaitement !)
def get_pair_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

# Test de la fonction
pairs_stats = get_pair_stats(vocab)
print("--- Statistiques des paires de lettres ---")
print(pairs_stats.most_common(5))  # Affiche les 5 paires les plus fréquentes