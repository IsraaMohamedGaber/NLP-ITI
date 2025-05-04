# NLP-ITI
NLP ITI Tasks Day by Day
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
- NumPy

Install dependencies using:

```bash
pip install gensim numpy
