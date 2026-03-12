# Transformer Seq2Seq

This project implements a **basic Transformer-based Seq2Seq model using PyTorch**.  
It was developed as part of an **exercise for understanding Large Language Models (LLMs)** and the internal architecture of Transformer encoder–decoder networks.

The model is trained on small example sentence pairs and demonstrates how transformers perform sequence-to-sequence text generation.

---

## Project Structure

```
project/
│
├── attention_masks.py      # Causal attention masking
├── encoder.py              # Transformer encoder
├── decoder.py              # Transformer decoder
├── transformer.py          # Seq2Seq transformer architecture
├── train.py                # Model training
├── inference.py            # Text generation
└── samples/                # Saved model + vocabulary
```

---

## Features

- Transformer Encoder–Decoder architecture
- Multi-head self-attention
- Causal masking for autoregressive decoding
- Small training dataset for demonstration
- Text generation using greedy decoding

---

## Installation

Install dependencies:

```bash
pip install torch
```

---

## Running the Project

### 1. Train the model

```
python train.py
```

This will train the transformer and generate:

```
samples/
   seq2seq.pth
   vocab.pkl
```

---

### 2. Run inference

```
python inference.py
```

Example:

```
Enter input: AI improves healthcare
Generated output: AI enhances medical diagnosis
```

---

## Learning Objectives

This exercise demonstrates:

- Transformer encoder–decoder architecture
- Attention masking
- Token embeddings
- Autoregressive sequence generation
- Basic LLM building blocks

---

## Technologies Used

- Python
- PyTorch
- Transformer Architecture

---

## Note

This implementation is **educational** and designed to illustrate the internal components of transformer-based language models rather than train a large-scale model.