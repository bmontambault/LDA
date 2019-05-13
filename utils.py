import numpy as np
import os

"""
This file defines functions for loading documents and transforming documents
into bag of words and word/document index representations
"""

def load_documents(files):
    """
    Load a list of files into a list of documents
        list files: list containing names of files to be loaded
    """
    docs = []
    for file in files:
        with open(file, 'r') as f:
            doc = f.read()
        docs.append(doc.split())
    return docs
    

def bag_of_words(docs, return_vocab=False):
    """
    Returns DxV matrix where D is the number of documents and V is the size
    of the vocabulary
        list docs: List of documents. Each document is a list of strings, 
                   where each string is a word
        bool return_vocab: Return vocab if True
    """
    vocab = list(set([a for b in docs for a in b]))
    X = np.array([[doc.count(w)/len(doc) for w in vocab] for doc in docs])
    if return_vocab:
        return X, vocab
    else:
        return X


def word_doc_indices(docs, return_vocab=False):
    """
    Returns Nx2 matrix where N is the total number of words across all documents
    and each row holds the word index and document index of that word
        list docs: List of documents. Each document is a list of strings, 
                   where each string is a word
        bool return_vocab: Return vocab if True
    """
    vocab = list(set([a for b in docs for a in b]))
    X = np.array([a for b in [[[vocab.index(w),i] for w in docs[i]] for i in range(len(docs))] for a in b])
    if return_vocab:
        return X, vocab
    else:
        return X