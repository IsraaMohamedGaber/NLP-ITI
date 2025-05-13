# NLP-ITI
NLP ITI Tasks Day by Day
# Day 1
# 🌐 Efficient Estimation of Word Representations in Vector Space

This repository contains an implementation and analysis of the Word2Vec models (Skip-Gram and CBOW) introduced by Tomas Mikolov et al. in their 2013 paper, "Efficient Estimation of Word Representations in Vector Space".

📘 Overview
The goal of this project is to learn high-quality vector representations (embeddings) of words from large text corpora. These embeddings capture meaningful syntactic and semantic relationships and can be used in various NLP tasks like classification, clustering, translation, and analogy solving.

We implement and evaluate two architectures in task of next day:

Continuous Bag-of-Words (CBOW)

Skip-Gram

# Day 2
# Word Embedding Prediction using Gensim

This project demonstrates how to use pretrained word embeddings from Gensim to perform word prediction using the **CBOW (Continuous Bag of Words)** and **Skip-gram** models.

We use the `glove-wiki-gigaword-100` model (100-dimensional GloVe vectors trained on Wikipedia and Gigaword) to:

- Predict a target word given context words (**CBOW**).
- Predict context words given a target word (**Skip-gram**).

---

## 📦 Dependencies

- Python 3.x
- [Gensim](https://radimrehurek.com/gensim/)
  
---
# Named Entity Recognition (NER) with spaCy using sample Data

This project demonstrates how to build a custom Named Entity Recognition (NER) model using spaCy. It uses a small dataset of labeled entities like organizations, people, and locations.

how to build ner model ?
1. dataset
2. framework that supports building ner models -> spacy
3. what is the shape of the dataset i need to make it work with spacy
4. build training pipeline
5. evaluate

---
# 🧠 Named Entity Recognition with spaCy & CoNLL2003

This project demonstrates how to train a **custom Named Entity Recognition (NER)** model using the **CoNLL2003** dataset and the **spaCy** NLP library. The CoNLL2003 dataset is a standard benchmark used in NLP to detect named entities like persons, organizations, locations, and more.

---

## 📌 Project Features

- Loads the [CoNLL2003](https://huggingface.co/datasets/conll2003) dataset automatically using the Hugging Face `datasets` library.
- Converts word-level tags into spaCy's required character-based annotation format.
- Trains a spaCy blank English model (`spacy.blank("en")`) from scratch.
- Evaluates the model performance on the validation set.
- Saves the trained model to disk for reuse.

---

# Day 5
# ROUGE-N Evaluation Metric (ROUGE-1)

## 📌 Overview

This project explains and demonstrates the use of **ROUGE-N**, a recall-based evaluation metric in **Natural Language Processing (NLP)**. ROUGE-N measures the overlap of n-grams between a candidate (generated) text and one or more human-written reference texts.

This README focuses on **ROUGE-1**, which evaluates **unigram (single word)** overlap and is commonly used for:

- Text summarization
- Machine translation
- Text generation
- Content comparison in NLP tasks

---

## 🧠 What is ROUGE-1?

**ROUGE-1** is a specific case of the ROUGE-N family where `N = 1`, meaning it looks at individual word matches (unigrams) between the reference and candidate text.

It is **recall-oriented**, meaning it evaluates how much of the reference content appears in the candidate output.

---

# Attention-Based Neural Machine Translation

## 📘 Overview

This project demonstrates the concept of **Attention Mechanisms** in **Neural Machine Translation (NMT)**, based on the influential paper  
**"Effective Approaches to Attention-based Neural Machine Translation"** by Luong, Pham, and Manning (2015).

The attention mechanism allows the model to focus on relevant parts of the input sentence while generating each word in the output, significantly improving translation quality—especially for long sentences.

---

## 🔍 What Is Attention in NMT?

Traditional encoder-decoder models compress the entire source sentence into a single fixed-length vector. This becomes a bottleneck, particularly with longer or more complex sentences.

**Attention** solves this by giving the decoder access to all encoder hidden states and learning to "attend" to the most relevant ones at each decoding step.

---



