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
# Named Entity Recognition (NER) with spaCy

This project demonstrates how to build a custom Named Entity Recognition (NER) model using spaCy. It uses a small dataset of labeled entities like organizations, people, and locations.

how to build ner model ?
1. dataset
2. framework that supports building ner models -> spacy
3. what is the shape of the dataset i need to make it work with spacy
4. build training pipeline
5. evaluate
