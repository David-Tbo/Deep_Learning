Oui. Pour un document pédagogique centré sur **LSTM, encoder-decoder, information bottleneck et attention**, je donnerais au Markdown le nom :

```text
sequence_to_sequence_attention.md
```

C'est suffisamment général et descriptif. Si ton arborescence est dédiée au deep learning, il s'intègre bien dans une section NLP.

Voici la version complète en **PyTorch**, en conservant la progression pédagogique de la réponse précédente.

````markdown
# Sequence-to-Sequence Models, LSTM and Attention

## 1. From text to numerical representation

A typical NLP pipeline can be summarized as:

```text
Raw text
   ↓
Tokenization
   ↓
Token IDs
   ↓
Padding / masking
   ↓
Embedding
   ↓
Neural network
   ↓
Prediction
````

---

## 2. Tokenization

Tokenization splits text into smaller units called **tokens**.

For example:

```text
"The cat is sleeping"
        ↓
["The", "cat", "is", "sleeping"]
```

Each token is then mapped to an integer ID using a vocabulary:

```text
"The"      → 2
"cat"      → 3
"is"       → 4
"sleeping" → 5
```

A simple vocabulary could be represented in Python as:

```python
word_to_id = {
    "<PAD>": 0,
    "<UNK>": 1,
    "the": 2,
    "cat": 3,
    "is": 4,
    "sleeping": 5
}
```

The text can then be converted into token IDs:

```python
sentence = ["the", "cat", "is", "sleeping"]

tokens = [word_to_id[word] for word in sentence]

print(tokens)
```

Output:

```text
[2, 3, 4, 5]
```

These integer IDs are a **discrete numerical representation** of the text.

> Tokenization converts text into tokens, and the vocabulary maps these tokens to integer IDs.

---

# 3. Embeddings

Neural networks cannot directly process word IDs as meaningful numerical values.

For example:

```text
cat → 3
dog → 7
```

The difference between `3` and `7` has no linguistic meaning.

An **embedding layer** therefore maps each token ID to a dense vector.

In PyTorch:

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=10_000,
    embedding_dim=128
)
```

If the input is:

```python
tokens = torch.tensor([2, 3, 4, 5])
```

then:

```python
vectors = embedding(tokens)

print(vectors.shape)
```

Output:

```text
torch.Size([4, 128])
```

Conceptually:

```text
Token ID
   ↓
Embedding
   ↓
Dense vector
```

For example:

```text
"cat" → 3 → [0.21, -0.17, 0.84, ...]
```

The embedding vectors are normally **learned during training**.

---

# 4. Padding

Sequences usually have different lengths.

For example:

```text
"I like NLP"

"I really like NLP"

"I like NLP very much"
```

Neural networks generally process batches of sequences with the same dimensions.

We therefore use **padding**:

```text
[12, 45, 78,  0,  0]
[12, 34, 45, 78,  0]
[12, 45, 78, 56, 91]
```

Here:

```text
0 = <PAD>
```

The zero does not represent a real word.

---

# 5. Ignoring padding with `padding_idx`

PyTorch provides a convenient mechanism in `nn.Embedding`:

```python
embedding = nn.Embedding(
    num_embeddings=num_tokens,
    embedding_dim=128,
    padding_idx=0
)
```

`padding_idx=0` tells the embedding layer that index `0` represents padding.

A useful comment is:

```python
padding_idx=0  # Identifies 0 as padding and prevents its embedding from being updated
```

This is slightly different from Keras' `mask_zero=True`.

With recurrent models, **masking and padding handling are distinct concepts**. `padding_idx=0` specifically handles the embedding associated with the padding token. If padded time steps must also be excluded from recurrent computation, PyTorch provides mechanisms such as packed sequences.

For example:

```python
from torch.nn.utils.rnn import pack_padded_sequence

packed = pack_padded_sequence(
    embeddings,
    lengths,
    batch_first=True,
    enforce_sorted=False
)

output, hidden = lstm(packed)
```

The key idea is:

> Padding is structural information, not linguistic information.

---

# 6. LSTM for sequence processing

An LSTM processes a sequence step by step while maintaining internal states.

Conceptually:

```text
x₁ → LSTM → h₁
x₂ → LSTM → h₂
x₃ → LSTM → h₃
x₄ → LSTM → h₄
```

In PyTorch:

```python
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    batch_first=True
)
```

Suppose the embedding layer produces:

```text
batch × sequence_length × embedding_dimension
```

For example:

```text
32 × 50 × 128
```

Then:

```python
embeddings = embedding(tokens)

output, (hidden, cell) = lstm(embeddings)
```

The output contains one hidden representation for each time step:

```text
h₁, h₂, h₃, ..., hₙ
```

---

# 7. Bidirectional LSTM

A bidirectional LSTM processes the sequence in both directions:

```text
Forward:

x₁ → x₂ → x₃ → x₄


Backward:

x₄ → x₃ → x₂ → x₁
```

In PyTorch:

```python
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    batch_first=True,
    bidirectional=True
)
```

The output dimension becomes:

```text
2 × hidden_size
```

because the forward and backward representations are concatenated.

Therefore:

```python
output.shape
```

will contain:

```text
batch × sequence_length × (2 × hidden_size)
```

---

# 8. Sequence-to-sequence architecture

A traditional sequence-to-sequence model contains two components:

```text
Input sequence
      ↓
   Encoder
      ↓
Context representation
      ↓
   Decoder
      ↓
Output sequence
```

The encoder processes the source sequence:

```text
x₁ → x₂ → x₃ → x₄
             ↓
      Encoder hidden states
```

The decoder then generates the target sequence.

---

# 9. The information bottleneck

Without attention, a traditional encoder-decoder architecture can be represented as:

```text
Input sequence
      ↓
    Encoder
      ↓
single context vector
      ↓
    Decoder
      ↓
Output sequence
```

For example:

```text
x₁ → x₂ → x₃ → x₄
             ↓
        context vector
             ↓
          Decoder
             ↓
      y₁ → y₂ → y₃
```

The encoder must compress the information from the entire input sequence into a **single fixed-size representation**.

This creates an **information bottleneck**.

For short sequences, this may work reasonably well.

For long sequences, however, the model has to compress a large amount of information into a single vector.

Some information may therefore be:

* forgotten;
* poorly represented;
* difficult to recover;
* less accessible when generating later outputs.

---

# 10. Encoder hidden states

Instead of considering only the final encoder state, we can retain all encoder hidden states:

```text
h₁   h₂   h₃   h₄
```

These states contain information about different parts of the input sequence.

For example:

```text
Input:

"The cat is sleeping"

       ↓

h₁    h₂    h₃    h₄
 ↓     ↓     ↓     ↓
The   cat    is sleeping
```

The attention mechanism allows the decoder to access these representations dynamically.

---

# 11. Why attention helps

Humans do not necessarily transform a sequence by compressing the entire source into a single representation.

For example, when translating a sentence, we continuously refer back to the source:

```text
Source sentence
      ↓
Focus on relevant words
      ↓
Generate current output word
      ↓
Focus on another part of the source
      ↓
Generate next output word
```

Attention implements a similar idea.

Instead of relying exclusively on one context vector, the decoder can access:

```text
h₁, h₂, h₃, ..., hₙ
```

and determine which states are most relevant at each generation step.

---

# 12. Attention weights

Suppose the encoder produces:

```text
h₁, h₂, h₃, h₄
```

and the decoder generates:

```text
s₀, s₁, s₂, ...
```

For every decoder time step, attention assigns a weight to every encoder hidden state.

We can write an attention weight as:

[
\alpha_{i,t}
]

The two indices are important.

### First index: `i`

`i` identifies the encoder hidden state:

[
h_i
]

For example:

```text
h₁ → i = 1
h₂ → i = 2
h₃ → i = 3
h₄ → i = 4
```

### Second index: `t`

`t` identifies the decoder time step:

[
s_t
]

For example:

```text
s₀ → t = 0
s₁ → t = 1
s₂ → t = 2
```

Therefore:

[
\alpha_{i,t}
]

means:

> The attention weight assigned to encoder hidden state (h_i) when the decoder is at time step (t).

For example:

[
\alpha_{3,0}
]

means:

> How much attention does the decoder pay to (h_3) when generating the output at decoder time step (s_0)?

---

# 13. Computing the context vector

The attention weights are used to construct a context vector.

The context vector at decoder time step (t) can be written:

[
c_t = \sum_i \alpha_{i,t} h_i
]

In other words:

```text
Encoder hidden states
        ↓
h₁   h₂   h₃   h₄
 \    |    |   /
  \   |    |  /
   Attention
       ↓
Weighted combination
       ↓
Context vector cₜ
       ↓
Decoder
       ↓
Output token
```

The context vector is therefore **different at each decoder time step**.

---

# 14. A simple PyTorch attention implementation

A simplified dot-product attention mechanism can be implemented as:

```python
import torch
import torch.nn.functional as F


def attention(query, keys, values):
    """
    Compute scaled dot-product attention.

    Parameters
    ----------
    query : Tensor
        Decoder representation.
    keys : Tensor
        Encoder hidden states used to compute attention scores.
    values : Tensor
        Encoder hidden states used to construct the context vector.

    Returns
    -------
    context : Tensor
        Weighted representation of the encoder states.
    weights : Tensor
        Attention weights.
    """

    # Compute attention scores
    scores = torch.bmm(
        query,
        keys.transpose(1, 2)
    )

    # Normalize scores into attention weights
    weights = F.softmax(scores, dim=-1)

    # Compute weighted sum of encoder states
    context = torch.bmm(weights, values)

    return context, weights
```

The important sequence is:

```text
Query
  ↓
Attention scores
  ↓
Softmax
  ↓
Attention weights
  ↓
Weighted sum of encoder states
  ↓
Context vector
```

---

# 15. Attention as a dynamic lookup mechanism

This is one of the most useful intuitions for understanding attention.

Without attention:

```text
Encoder
h₁ → h₂ → h₃ → h₄
                 ↓
          single context
                 ↓
              Decoder
```

With attention:

```text
             ┌── h₁
             ├── h₂
Decoder ─────┼── h₃
             └── h₄
                ↓
        attention weights
                ↓
          context vector
                ↓
             output
```

And this process is repeated for every decoder time step.

Therefore, the decoder can effectively ask:

> **Which parts of the input are most relevant for generating the next output?**

---

# 16. Attention reduces the information bottleneck

Without attention:

```text
Entire input
     ↓
single fixed-size representation
     ↓
decoder
```

With attention:

```text
Entire input
     ↓
h₁, h₂, ..., hₙ
     ↓
attention at each decoder step
     ↓
relevant weighted combination
     ↓
decoder
```

Therefore:

> **Attention reduces the information bottleneck by allowing the decoder to dynamically access the encoder's hidden states instead of relying only on a single fixed context vector.**

---

# 17. From LSTM + Attention to Transformers

The historical progression can be viewed approximately as:

```text
RNN
 ↓
LSTM / GRU
 ↓
Encoder–Decoder LSTM
 ↓
Encoder–Decoder LSTM + Attention
 ↓
Transformer
 ↓
GPT and modern language models
```

The Transformer architecture made **attention the central mechanism** instead of relying on recurrent processing.

GPT models are based on **autoregressive Transformers**.

During text generation, the model predicts the next token from the preceding context:

```text
The cat is
      ↓
    model
      ↓
sleeping
```

Then:

```text
The cat is sleeping
      ↓
    model
      ↓
...
```

---

# 18. LSTM versus Transformer

A useful high-level comparison is:

|                         | LSTM                     | Transformer                 |
| ----------------------- | ------------------------ | --------------------------- |
| Main mechanism          | Recurrence               | Attention                   |
| Processes sequence      | Sequentially             | In parallel during training |
| Long-range dependencies | More difficult           | More easily modeled         |
| Context access          | Through recurrent states | Through attention           |
| Attention               | Optional                 | Central mechanism           |
| Example                 | Seq2Seq LSTM             | GPT                         |

The important distinction is not that an LSTM "cannot understand" long sequences, but that recurrent architectures must propagate information through a sequence of recurrent states.

Attention provides a more direct path between different positions in the sequence.

---

# 19. Early stopping in PyTorch

Unlike Keras, PyTorch does not provide a standard built-in `EarlyStopping` callback in its core API.

The logic is usually implemented manually.

For example:

```python
best_val_loss = float("inf")
patience = 3
epochs_without_improvement = 0

for epoch in range(num_epochs):

    # Training
    model.train()

    # ...

    # Validation
    model.eval()

    # Compute validation loss
    val_loss = ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "best_model.pt"
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping")
        break
```

Here:

```text
patience = 3
```

means that training stops after **3 consecutive epochs without improvement** in validation loss.

---

# 20. Saving and loading a PyTorch model

For PyTorch models, it is common to save the model parameters:

```python
torch.save(
    model.state_dict(),
    "model.pt"
)
```

The model architecture must then be recreated:

```python
model = MyModel(...)

model.load_state_dict(
    torch.load("model.pt")
)

model.eval()
```

---

# 21. The complete mental model

For sequence generation, remember these three levels:

```text
TOKENIZATION
"What is the text made of?"
        ↓
Token IDs


EMBEDDING
"How can each token be represented numerically?"
        ↓
Dense vectors


SEQUENCE MODEL
"How do the tokens interact across the sequence?"
        ↓
LSTM / Transformer


ATTENTION
"Which parts of the context are relevant right now?"
        ↓
Context-dependent representation
```

For an encoder-decoder LSTM with attention:

```text
                 ENCODER
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
         h₁        h₂   ...   hₙ
          │         │         │
          └─────────┼─────────┘
                    ↓
                ATTENTION
                    ↑
                    │
             Decoder state
                    │
                    ↓
             Context vector
                    │
                    ↓
                 DECODER
                    │
                    ↓
             Next output token
```

The central idea is:

> **Tokenization converts text into token IDs, embeddings convert token IDs into dense vectors, LSTMs process sequences through recurrent states, and attention allows the decoder to dynamically focus on the most relevant encoder states when generating each output token.**

````

### Nom du fichier

Je recommande donc :

```text
sequence_to_sequence_attention.md
````

Si tu veux que ce document fasse partie d'un ensemble plus large consacré aux architectures NLP, une structure cohérente serait par exemple :

```text
deep_learning/
└── NLP/
    ├── tokenization.md
    ├── embeddings.md
    ├── recurrent_neural_networks.md
    ├── lstm.md
    ├── sequence_to_sequence.md
    ├── sequence_to_sequence_attention.md
    └── transformers.md
```

