# 📝 Cheatsheet: Byte-Pair Encoding (BPE)

**Core Goal:** Eliminate the `[UNK]` (Unknown) token. BPE builds a vocabulary of **subwords** by iteratively merging the most frequent pairs of characters/symbols. This allows LLMs to handle rare words, typos, and unseen words by breaking them into known building blocks.

---

## 🚀 1. The Core Philosophy

```
Traditional (Word-level)  👉  "lower" ❌ Unseen Word  👉  [UNK] (Zero meaning)
BPE (Subword-level)       👉  "lower" 📦 Systematic   👉  ["low", "er"] (Retains meaning)

```

1. **Tokenization is upfront:** The text is sliced into subwords **before** the neural network processes it.
2. **Deterministic rules:** The tokenizer uses a fixed, ordered list of **Merge Rules** learned during training.

---

## 🔍 2. The Toy Example (Step-by-Step)

### Base Setup

* **Initial Vocabulary** (words split into characters + end-of-word symbol `</w>`):
`{"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}`

### The Merge Iterations (Training)

Every loop finds the most frequent adjacent bigram and fuses it:

| Iteration | Most Frequent Pair | Count | Resulting Vocabulary Update |
| --- | --- | --- | --- |
| **Merge 1** | `('e', 's')` | 9 | `e s` $\rightarrow$ `es` (found in *newest* and *widest*) |
| **Merge 2** | `('es', 't')` | 9 | `es t` $\rightarrow$ `est` |
| **Merge 3** | `('l', 'o')` | 7 | `l o` $\rightarrow$ `lo` (found in *low* and *lower*) |
| **Merge 4** | `('lo', 'w')` | 7 | `lo w` $\rightarrow$ `low` |
| **Merge 5** | `('e', 'r')` | 2 | `e r` $\rightarrow$ `er` |

### The Artifact: Learned Merge Rules (Ordered!)

1. `e` + `s` $\rightarrow$ `es`
2. `es` + `t` $\rightarrow$ `est`
3. `l` + `o` $\rightarrow$ `lo`
4. `lo` + `w` $\rightarrow$ `low`
5. `e` + `r` $\rightarrow$ `er`

---

## ⚙️ 3. Inference Mechanism (How Unseen Words are Sliced)

When a user inputs the unseen word **`"lower"`** at inference time, it goes through the "meat grinder" of ordered merge rules:

```
[ l, o, w, e, r ]     Rule 1 ('e'+'s') ? ❌ No
[ l, o, w, e, r ]     Rule 2 ('es'+'t')? ❌ No
[ lo, w, e, r ]       Rule 3 ('l'+'o') ?  Match! -> merged to 'lo'
[ low, e, r ]         Rule 4 ('lo'+'w')?  Match! -> merged to 'low'
[ low, er ]           Rule 5 ('e'+'r') ?  Match! -> merged to 'er'

```

**Final Output Tokens Passed to LLM:** `["low", "er"]`

---

## 🎯 4. Three Massive LLM Benefits

* **Zero `[UNK]` issues:** Worst-case scenario for a completely alien word? It gets broken down into individual fallback characters (`['a', 'b', 'c']`), but never fails.
* **Morphological Understanding:** The Transformer captures semantics because `low` links to the base concept, and `er` links to the comparative modifier vector.
* **Compact Vocabulary Size:** Instead of storing millions of whole words (including all conjugations/plurals), the model stores a compact vocabulary (~32k to 100k tokens) capable of assembling *any* word.