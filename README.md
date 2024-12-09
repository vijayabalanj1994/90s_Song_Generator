# 90s_Song_Generator

**Crafting a 90s Song Generator: Building Language Models with NLTK and PyTorch**

This project explores the creation of language models to generate songs with a 90s vibe. It leverages bigram, 4-gram, and 8-gram models using statistical and neural network approaches, implemented with Python libraries such as NLTK, PyTorch, and Matplotlib.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Implementation Details](#implementation-details)
  - [Data Preprocessing](#data-preprocessing)
  - [Bigram Model](#bigram-model)
  - [Feedforward Neural Network Models](#feedforward-neural-network-models)
  - [Training](#training)
  - [Visualizations](#visualizations)
- [Results](#results)
  - [Generated Songs](#generated-songs)
  - [Loss and Perplexity Analysis](#loss-and-perplexity-analysis)
- [Conclusion](#conclusion)

---

## Introduction

The **90s_Song_Generator** is a project designed to recreate the essence of 90s songs through a combination of statistical and neural language models. This project showcases the application of Natural Language Processing (NLP) and Machine Learning (ML) techniques to generate new song lyrics inspired by an iconic era.

---

## Features

- **Statistical Bigram Model**: Computes conditional probabilities for word prediction.
- **Feedforward Neural Networks (FNNs)**: Implements bigram, 4-gram, and 8-gram models for next-word prediction.
- **Word Embedding Visualization**: Uses t-SNE to project high-dimensional word embeddings into a 2D space.
- **Song Generation**: Generates lyrics based on trained models.
- **Loss and Perplexity Analysis**: Tracks model performance using cross-entropy loss and perplexity metrics.

---

## Dependencies

This project requires the following libraries:

- Python 3.8+
- NLTK
- PyTorch
- Matplotlib
- Scikit-learn
- NumPy
- Pandas

Install dependencies using:

```bash
pip install nltk torch matplotlib scikit-learn numpy pandas
```

---

## Implementation Details

### Data Preprocessing
- **Tokenization**: Tokenizes the input text into words using NLTK.
- **Vocabulary Creation**: Builds a vocabulary with mappings between words and indices.
- **N-Gram Construction**: Constructs n-grams (e.g., bigrams, 4-grams, etc.) for context-target pairs.

### Bigram Model
The statistical bigram model calculates conditional probabilities using token frequencies to predict the next word given a context.

### Feedforward Neural Network Models
Three neural models are implemented:
1. **Bigram (2-gram) Model**
2. **4-gram Model**
3. **8-gram Model**

Each model includes:
- Embedding layer for word embeddings.
- Two fully connected layers with ReLU activation.
- Output layer for next-word prediction.

### Training
- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Stochastic Gradient Descent (SGD).
- **Learning Rate Scheduler**: Reduces the learning rate over time.
- **Metrics**: Tracks cross-entropy loss and perplexity across epochs.

### Visualizations
- **Word Embeddings**: Uses t-SNE to visualize word similarity.
- **Loss and Perplexity**: Plots cross-entropy loss and perplexity for each model.

---

## Results

### Generated Songs

#### Bigram Model
```
We are no strangers to love you cry never gonna tell a lie and hurt you...
```

#### 4-gram Model
```
We are no strangers to love you know the rules and were you never gonna give...
```

#### 8-gram Model
```
We are no strangers to love You know the rules let you down never...
```

### Loss and Perplexity Analysis
The cross-entropy loss and perplexity are plotted for each model to evaluate performance over epochs.

---

## Conclusion

This project demonstrates how statistical and neural approaches can be combined to generate creative content. By leveraging word embeddings, n-grams, and deep learning models, the **90s_Song_Generator** replicates the lyrical style of 90s hits.

Feel free to contribute or share your suggestions to improve this project!
