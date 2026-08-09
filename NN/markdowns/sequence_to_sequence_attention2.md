# Key Takeaways from the Seq2Seq NMT Discussion

## 1. Initial model

The starting point was a **Hungarian → English Neural Machine Translation (NMT)** model implemented with TensorFlow/Keras.

The architecture was:

```text
Source sentence
      ↓
Embedding
      ↓
Encoder LSTM
      ↓
Final hidden/cell states
      ↓
Decoder LSTM
      ↓
Dense + Softmax
      ↓
Target sentence
```

The model used:

* LSTM encoder/decoder
* 512-dimensional embeddings
* 512 hidden units
* Dropout = 0.2
* Batch size = 256
* Adam optimizer
* Sparse categorical cross-entropy
* Teacher forcing
* `<sos>` and `<eos>` target tokens

The initial model **did not use attention**.

---

## 2. Initial performance

The initial training results were approximately:

```text
loss:                       4.506
sparse_categorical_accuracy: 0.0695
val_loss:                   3.076
val_accuracy:               0.1026
```

However, the model was trained for only **one epoch**.

Therefore, these results should not be interpreted as the final performance of the architecture. The training duration was clearly insufficient to assess convergence.

---

## 3. First optimization: Adam learning rate

The default Keras Adam optimizer uses a learning rate of approximately:

```python
0.001
```

A lower learning rate can be tested for Seq2Seq training:

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005
)
```

Then:

```python
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)
```

The learning rate should be treated as a **hyperparameter**, not as a universally optimal value.

---

## 4. Important architectural improvement: attention

The original architecture compresses the complete source sentence into the final encoder states:

```text
Source sentence
      ↓
Encoder LSTM
      ↓
Final state
      ↓
Decoder
```

This creates a bottleneck, especially for long sentences.

The proposed improvement was to introduce **Bahdanau attention**.

Conceptually:

```text
                ┌──────────────────────┐
Source tokens → │    Encoder LSTM      │
                └──────────┬───────────┘
                           │
                  all encoder outputs
                           │
                           ▼
                    ┌─────────────┐
Decoder state ─────►│  Attention  │
                    └──────┬──────┘
                           │
                     Context vector
                           │
                           ▼
                    Decoder LSTM
                           │
                           ▼
                    Target tokens
```

The important idea is that the decoder should not rely exclusively on the final encoder state.

At each decoding step, attention determines which source positions are most relevant.

---

## 5. Problem identified in the Keras implementation

The proposed Keras attention implementation contained a more fundamental architectural issue than simply the `state_h` dimension.

The encoder was originally defined as:

```python
encoder_lstm = layers.LSTM(
    hidden_dim,
    return_state=True,
    dropout=default_dropout
)
```

This means:

```python
encoder_outputs, state_h, state_c = encoder_lstm(...)
```

but `encoder_outputs` is only the **final output** of the LSTM.

Its shape is approximately:

```text
(batch_size, hidden_dim)
```

For Bahdanau attention, we need the **sequence of encoder outputs**:

```text
(batch_size, source_sequence_length, hidden_dim)
```

Therefore, the encoder must use:

```python
return_sequences=True
```

For example:

```python
encoder_lstm = layers.LSTM(
    hidden_dim,
    return_sequences=True,
    return_state=True,
    dropout=default_dropout
)
```

Then:

```python
encoder_outputs, state_h, state_c = encoder_lstm(
    encoder_embedding_output
)
```

produces:

```text
encoder_outputs → (batch, source_length, hidden_dim)
state_h        → (batch, hidden_dim)
state_c        → (batch, hidden_dim)
```

This is the correct structure for attention.

---

## 6. Another problem in the proposed implementation

The code computed:

```python
context_vector, attention_weights = attention(
    encoder_outputs,
    state_h
)
```

but then calculated:

```python
decoder_lstm_input = tf.concat(
    [context_vector, decoder_embedding_output],
    axis=-1
)
```

while actually calling:

```python
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding_output,
    initial_state=encoder_states
)
```

Therefore, the calculated `decoder_lstm_input` was **never used**.

This is a critical issue.

The attention mechanism was therefore not actually influencing the decoder.

---

## 7. Why the proposed `tf.expand_dims(state_h, 1)` was not the correct solution

The previous discussion suggested:

```python
attention(encoder_outputs, tf.expand_dims(state_h, 1))
```

This is not the correct fix.

The `BahdanauAttention` implementation itself already performs:

```python
hidden_with_time_axis = tf.expand_dims(hidden_state, 1)
```

Therefore, passing:

```python
state_h
```

is correct.

The real problem was that `encoder_outputs` did not contain the sequence of encoder states because the encoder lacked:

```python
return_sequences=True
```

The correct conceptual dimensions are:

```text
encoder_outputs:
(batch, source_length, hidden_dim)

state_h:
(batch, hidden_dim)
```

and the attention layer internally converts the latter into:

```text
(batch, 1, hidden_dim)
```

---

# 8. Most important architectural conclusion

A proper attention-based Seq2Seq architecture should therefore look like:

```text
                 ENCODER
                   
Source tokens
     │
     ▼
Embedding
     │
     ▼
LSTM
     │
     ├──────────────► Encoder outputs
     │                (all time steps)
     │
     └──────────────► Final states
                         │
                         ▼
                    DECODER
                         │
              ┌──────────┴──────────┐
              │                     │
       Decoder state          Encoder outputs
              │                     │
              └──────────┬──────────┘
                         ▼
                     Attention
                         │
                         ▼
                   Context vector
                         │
                         ▼
              Decoder LSTM + context
                         │
                         ▼
                    Softmax
                         │
                         ▼
                 Target sentence
```

---

# 9. Training improvements

Several training improvements were identified.

### Increase the number of epochs

The original experiment used:

```python
epochs = 1
```

which is insufficient to evaluate the model.

A better approach is to allow several epochs and use:

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

The exact number of epochs should be determined by validation performance.

### Use ReduceLROnPlateau

The code defined:

```python
lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=3,
    min_lr=1e-6
)
```

but the callback was not included in the actual `model.fit()` call.

It should be included if this strategy is retained:

```python
callbacks=[
    cp_callback,
    es_callback,
    lr_callback
]
```

---

# 10. Other potential improvements

The following improvements should be considered after fixing the architecture:

1. **Bahdanau attention**
2. **Bidirectional encoder**
3. **Stacked LSTM layers**
4. **Better tokenization**
5. **Subword tokenization**
6. **Gradient clipping**
7. **Learning-rate scheduling**
8. **Teacher-forcing strategy**
9. **Beam search during inference**
10. **BLEU / chrF evaluation rather than token accuracy alone**

In particular, **token-level accuracy is not a very good primary metric for machine translation**.

BLEU, chrF, and eventually COMET are more informative for translation quality.

---

# 11. Major implementation decision: move from Keras to PyTorch

The most important conclusion from this discussion is:

> **The Keras implementation should ultimately be replaced by a PyTorch implementation.**

The target architecture should remain the same conceptually:

```text
Hungarian
   ↓
Tokenizer
   ↓
Embedding
   ↓
Encoder LSTM
   ↓
Encoder hidden states
   ↓
Bahdanau Attention
   ↓
Decoder LSTM
   ↓
Linear layer
   ↓
Softmax
   ↓
English
```

The migration should not simply reproduce the existing Keras code line-by-line.

Instead, the PyTorch implementation should be redesigned around explicit:

```python
Encoder
Attention
Decoder
Seq2Seq
```

classes.

A clean project structure would be:

```text
nmt/
├── data/
│   ├── hun_eng_pairs_train.txt
│   └── hun_eng_pairs_val.txt
│
├── preprocessing/
│   └── tokenizer.py
│
├── models/
│   ├── encoder.py
│   ├── attention.py
│   ├── decoder.py
│   └── seq2seq.py
│
├── train.py
├── evaluate.py
└── inference.py
```

---

# 12. Recommended PyTorch architecture

The main components should be separated.

### Encoder

```python
class Encoder(nn.Module):
    ...
```

Responsible for:

* source embeddings
* LSTM
* encoder outputs
* final hidden state
* final cell state

### Bahdanau Attention

```python
class BahdanauAttention(nn.Module):
    ...
```

Responsible for:

* comparing decoder hidden state with encoder outputs
* calculating attention scores
* applying softmax
* producing the context vector

### Decoder

```python
class Decoder(nn.Module):
    ...
```

Responsible for:

* target embeddings
* attention
* LSTM
* output projection

### Seq2Seq

```python
class Seq2Seq(nn.Module):
    ...
```

Responsible for connecting:

```text
Encoder → Attention → Decoder
```

and implementing teacher forcing.

---

# 13. Recommended next step

The next implementation should therefore **not be another incremental modification of the current Keras code**.

The cleaner approach is to rebuild the model in PyTorch with:

```text
Encoder LSTM
      +
Bahdanau Attention
      +
Decoder LSTM
      +
Teacher Forcing
      +
Gradient Clipping
      +
Adam
      +
Learning-rate scheduler
```

while keeping the existing Hungarian-English dataset and preprocessing logic as much as possible.

This will also make the attention mechanism much easier to inspect and debug because the tensors and decoding loop are explicit.

---

# Final takeaway

The main lessons from the discussion are:

1. The original model is a **Seq2Seq LSTM without attention**.
2. One training epoch is insufficient to judge its performance.
3. Adam can be configured explicitly with:

   ```python
   Adam(learning_rate=0.0005)
   ```
4. Bahdanau attention requires **all encoder time-step outputs**.
5. Therefore the encoder must use:

   ```python
   return_sequences=True
   ```
6. `state_h` should be passed to the attention layer without manually adding a dimension because the attention layer does that internally.
7. The previously calculated `decoder_lstm_input` was not actually passed to the decoder, so the proposed attention implementation was incomplete.
8. Translation quality should be evaluated with metrics such as **BLEU and chrF**, not only token accuracy.
9. The next version should be **implemented in PyTorch**, with separate `Encoder`, `BahdanauAttention`, `Decoder`, and `Seq2Seq` classes.
10. The PyTorch version should be treated as a clean architectural rewrite rather than a direct line-by-line translation of the Keras implementation.
