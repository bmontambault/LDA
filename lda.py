import numpy as np

from utils import word_doc_indices

"""
This file defines the LDA model class
"""    

class LDA:
    """
    Latent Dirichlet Allocation
    """
    def __init__(self, vocab=None):
        self.vocab = vocab
        
    
    def get_vocab(self, docs, return_data=False):
        """
        Get list of unique words from a set of documents
            list docs: List of documents. Each document is a list of strings, 
                       where each string is a word
            bool return_data: Return transformed data if True
        """
        X, self.vocab = word_doc_indices(docs)
        if return_data:
            return X
        
    
    def fit_raw(self, docs, K, alpha, beta, niters):
        """
        Fit LDA from list of documents
            list docs: List of documents. Each document is a list of strings, 
                       where each string is a word
            int K: Number of topics
            float alpha: Dirichlet parameter for topic distribution
            float beta: Dirichlet parameter for word distribution
            int niters: Number of iterations for collapsed Gibbs sampler
        """
        X = self.get_vocab(docs, True)
        self.fit(X, K, alpha, beta, niters)
        
    
    def fit(self, X, K, alpha, beta, niters):
        """
        Fit LDA from words/indices representation of documents using collapsed
        Gibbs samplers
            array X: Words/indices representation of documents
            int K: Number of topics
            float alpha: Dirichlet parameter for topic distribution
            float beta: Dirichlet parameter for word distribution
            int niters: Number of iterations for collapsed Gibbs sampler
        """
        V, D = X.max(axis=0)+1
        if self.vocab == None:
            self.vocab = list(range(V))
        
        N = len(X)
        Z = np.random.randint(0, K, N)
        C_d = np.zeros((D,K))
        C_t = np.zeros((K,V))
                
        for i in range(N):
            topic = Z[i]
            word, doc = X[i]
            C_d[doc][topic] += 1
            C_t[topic][word] += 1
        
        P = np.zeros(K)
        permutation = np.random.permutation(np.arange(N))
        
        for i in range(niters):
            for j in permutation:
                word, doc = X[j]
                topic = Z[j]
                C_d[doc][topic] -= 1
                C_t[topic][word] -= 1
                
                for k in range(K):
                    p_t = (C_t[k][word]+beta)/(V*beta + C_t[k].sum())
                    p_d = (C_d[doc][k]+alpha)/(K*alpha + C_d[doc].sum())
                    P[k] = p_t * p_d
                P = P/P.sum()
                
                new_topic = np.random.choice(np.arange(K), p=P)
                Z[j] = new_topic
                C_d[doc][new_topic] +=1
                C_t[new_topic][word] += 1
        
        self.topics = np.flip(np.array(self.vocab)[C_t.argsort(axis=1)], axis=1)
        self.docs_repr = (alpha + C_d) / (K*alpha + C_d.sum(axis=1)[:,None])