# Neural Machine Translation — Key Takeaways

## 1. Encoder–Decoder architecture

A classical NMT model can be divided into two components:

* **Encoder**: reads the source sentence and converts it into hidden states representing its context.
* **Decoder**: generates the target sentence token by token from these states.

For an LSTM decoder, the relevant recurrent states are:

* `h` — hidden state
* `c` — cell state

The encoder passes these states to the decoder as its initial state.

---

## 2. Teacher forcing

During training, the decoder can use **teacher forcing**:

> At time step `t`, the decoder receives the **true target token from time step `t-1`** rather than its own previous prediction.

For example:

```text
Target sentence:

<bos>   I       am      fine   <eos>
          ↓       ↓       ↓
Decoder:  I      am     fine   <eos>
```

During training:

```text
Input to decoder:   <bos>   I       am      fine
Expected output:     I      am      fine    <eos>
```

This makes training easier and faster.

### The important consequence

During inference, the correct previous token is **not available**.

Therefore:

```text
Training:
true previous token → decoder → next token

Inference:
predicted previous token → decoder → next token
```

This difference between training and inference is known as **exposure bias**.

---

# 3. Why inference is performed step by step

During inference, the decoder must operate autoregressively:

```text
Source sentence
      │
      ▼
   Encoder
      │
      ▼
Initial decoder states
      │
      ▼
   Decoder
      │
      ▼
 first predicted token
      │
      ▼
   Decoder
      │
      ▼
 second predicted token
      │
      ▼
     ...
```

At each step:

1. Feed the current token to the decoder.
2. Obtain the probability distribution for the next token.
3. Select a token.
4. Pass this predicted token to the next decoder step.
5. Pass the new hidden states forward.

This is why an inference-time decoder is often implemented differently from the training graph.

---

# 4. Decoder inputs and outputs

A decoder based on an LSTM can be represented as:

```text
token_t
   │
   ▼
Embedding
   │
   ▼
LSTM
 ┌─┴─────────────┐
 ▼               ▼
output_t       h_t, c_t
 │
 ▼
Linear layer
 │
 ▼
Softmax
 │
 ▼
P(token_{t+1})
```

The important point is that the decoder has **two types of information flowing through it**:

### Token information

```text
token_t → Embedding → LSTM
```

### Recurrent state information

```text
(h_t, c_t) → next decoder step
```

The hidden states are therefore not simply a "vector containing the sentence". They are the internal recurrent state used to maintain contextual information.

---

# 5. `get_layer()` and model reconstruction

The earlier Keras example used:

```python
model.get_layer(...)
```

to retrieve trained layers and reconstruct separate encoder and decoder models.

Conceptually, however, the important idea is not Keras itself:

> **The inference models reuse the weights learned during training.**

There is no second training phase.

In PyTorch, this is generally easier to express explicitly because the encoder and decoder are normally defined as separate `nn.Module` classes.

For example:

```python
encoder = Encoder(...)
decoder = Decoder(...)
```

The training model combines them, but during inference they can naturally be called separately.

---

# 6. PyTorch is preferable for this educational implementation

For a from-scratch NMT implementation, **PyTorch is preferable to Keras** because it makes the mechanics of the encoder–decoder architecture more explicit.

A typical structure is:

```text
NMT
├── Encoder
│   ├── Embedding
│   └── LSTM
│
├── Decoder
│   ├── Embedding
│   ├── LSTM
│   └── Linear
│
└── Training / Inference loop
```

This makes it particularly clear that:

* the encoder produces the initial states;
* the decoder consumes a token and previous states;
* the decoder produces a prediction and new states;
* the prediction becomes the next decoder input during inference.

---

# 7. Text preprocessing

A preprocessing function such as:

```python
def preprocess_sentence(s):
    s = normalize_unicode(s)
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = s.strip()
    return s
```

has several purposes:

1. Normalize Unicode.
2. Separate punctuation from words.
3. Normalize multiple spaces.
4. Remove leading/trailing whitespace.

For example:

```text
Hello,world!
```

becomes approximately:

```text
Hello , world !
```

This makes tokenization easier.

---

# 8. Should punctuation be removed?

For NMT, **generally no**.

Punctuation carries syntactic and sometimes semantic information.

For example:

```text
Let's eat, grandma!
```

is different from:

```text
Let's eat grandma!
```

Therefore, rather than deleting punctuation, it is generally preferable to:

* normalize it;
* tokenize it consistently;
* let the model learn how to reproduce it.

Removing final punctuation should therefore be done only if there is a specific reason in the target application.

---

# 9. `rstrip()` when reading the dataset

Given:

```python
with open(file_name) as file:
    train = [line.rstrip() for line in file]
```

`rstrip()` removes characters from the **right-hand side** of each line.

Most importantly, it removes the newline:

```text
"Hello world!\n"
        ↓
"Hello world!"
```

It can also remove trailing spaces.

It does **not** remove characters from the beginning of the string.

---

# 10. Separating source and target sentences

If the dataset contains:

```text
Teszek rá, mit mondasz!<sep>I don't care what you say.
```

then:

```python
sentence.split("<sep>")
```

produces:

```python
[
    "Teszek rá, mit mondasz!",
    "I don't care what you say."
]
```

However:

```python
sentence1 = ["..."]
sentence1.split("<sep>")
```

does not work because `sentence1` is a **list**, not a string.

You need:

```python
sentence1[0].split("<sep>")
```

For an entire dataset:

```python
train_input, train_target = zip(
    *[pair.split("<sep>") for pair in train]
)
```

---

# 11. Tokenization

A tokenizer converts text into integer token IDs.

For example:

```text
"hello how are you"
```

might become:

```text
[15, 27, 8, 42]
```

The model does not directly process words. It processes these integer IDs.

With PyTorch, a typical vocabulary is explicitly constructed:

```python
vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    ...
}
```

This is preferable for understanding what is happening internally.

---

# 12. Special tokens

A useful NMT vocabulary normally contains at least:

| Token   | Purpose                              |
| ------- | ------------------------------------ |
| `<pad>` | Padding sequences to the same length |
| `<unk>` | Unknown/out-of-vocabulary token      |
| `<bos>` | Beginning of sentence                |
| `<eos>` | End of sentence                      |

The decoder uses:

```text
<bos> → first prediction → ... → <eos>
```

This is especially important during inference because the decoder needs to know **when to start and when to stop**.

---

# 13. Why vocabulary size often includes `+1`

With the old Keras `Tokenizer`, word IDs start at `1`:

```text
word → 1
word → 2
word → 3
...
```

while `0` is commonly reserved for padding.

Therefore:

```python
vocab_size = len(tokenizer.word_index) + 1
```

ensures that index `0` is also included in the vocabulary size.

With PyTorch, it is better to make this explicit:

```python
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
```

and construct the vocabulary accordingly.

This avoids relying on implicit conventions.

---

# 14. A complete conceptual picture

The most important workflow to retain is:

```text
                 TRAINING
                 ────────

Hungarian sentence
        │
        ▼
     Encoder
        │
        ├──── h
        └──── c
             │
             ▼
       Decoder + teacher forcing
             │
             ▼
      English predictions
```

Where the decoder receives the **true previous target token**.

During inference:

```text
                 INFERENCE
                 ─────────

Hungarian sentence
        │
        ▼
     Encoder
        │
        ├──── h
        └──── c
             │
             ▼
       Decoder
          ▲
          │
      <bos>
          │
          ▼
     prediction 1
          │
          ▼
     prediction 2
          │
          ▼
        ...
          │
          ▼
       <eos>
```

The crucial difference is:

```text
TRAINING
true target token → decoder

INFERENCE
model prediction → decoder
```

That is the central concept behind the encoder–decoder NMT architecture with teacher forcing.
