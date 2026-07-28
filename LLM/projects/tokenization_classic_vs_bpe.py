# Classical word tokenization
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Unbelievable tokenization works differently.")
print([t.text for t in doc])
vocabs = [t.text for t in doc]
print(vocabs)

# Byte-Pair Encoding (BPE) — the algorithm behind GPT-style tokenizers
from collections import Counter

def get_pair_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    merged = "".join(pair)
    new_vocab = {}
    bigram = " ".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, merged)
        new_vocab[new_word] = freq
    return new_vocab

# Toy vocabulary: words represented as space-separated characters + end token
vocab = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}

for i in range(5):
    pairs = get_pair_stats(vocab)
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    print(f"Merge {i+1}: {best_pair}")