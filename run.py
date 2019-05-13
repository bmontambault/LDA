import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_documents, bag_of_words, word_doc_indices
from models import LogisticRegression
from lda import LDA

"""
This file trains the LDA model (task 1) and outputs error rates and plots
for the bag of words and LDA representations on the classification task (task 2)
"""


if __name__ == "__main__":
    
    files = [str(i) for i in range(1, 201)]
    docs = load_documents(files)
    X, vocab = word_doc_indices(docs, return_vocab=True)
    bow_X = bag_of_words(docs, return_vocab=False)
    
    print ("Task 1")
    lda = LDA(vocab)
    lda.fit(X, 20, .1, .01, 100)
    
    i = 0
    for topics in lda.topics[:,:5]:
        print ('Topic '+str(i+1) + ': ' + ', '.join(topics))
        i+=1
    
    print ("Writing topics to topicwords.csv")
    pd.DataFrame(lda.topics[:,:5]).to_csv('topicwords.csv', header=None, index=False)
    
    
    print("Task 2")
    y = pd.read_csv('index.csv', header=None).values[:,1].ravel()
    y = y[np.array(files).astype(int)-1]
    
    
    bow_lr = LogisticRegression()
    bow_lr.learning_curve(bow_X, y, args={'maximize_evidence':True}, other_params=['alpha'],
                      add_dim=True, runs=30, m=10, test_size=.4, max_train_size=80)
    
    print ('Bag of Words Mean Error Rates: ' + ', '.join([str(x) for x in bow_lr.learning_curve_mean]))
    
    lda_lr = LogisticRegression()
    lda_lr.learning_curve(lda.docs_repr, y, args={'maximize_evidence':True}, other_params=['alpha'],
                      add_dim=True, runs=30, m=10, test_size=.4, max_train_size=80)
    
    print ('LDA Mean Error Rates: ' + ', '.join([str(x) for x in lda_lr.learning_curve_mean]))
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(bow_lr.learning_curve_sizes, bow_lr.learning_curve_mean, label='Bag of Words')
    bow_lr_lower = bow_lr.learning_curve_mean - bow_lr.learning_curve_std
    bow_lr_upper = bow_lr.learning_curve_mean + bow_lr.learning_curve_std
    ax.fill_between(bow_lr.learning_curve_sizes, bow_lr_lower, bow_lr_upper, alpha=.2)
    
    ax.plot(lda_lr.learning_curve_sizes, lda_lr.learning_curve_mean, label='LDA')
    lda_lr_lower = lda_lr.learning_curve_mean - lda_lr.learning_curve_std
    lda_lr_upper = lda_lr.learning_curve_mean + lda_lr.learning_curve_std
    ax.fill_between(lda_lr.learning_curve_sizes, lda_lr_lower, lda_lr_upper, alpha=.2)
            
    ax.set_xlabel('Train Size')
    ax.set_ylabel('Average Error Rate')
    plt.legend(loc='upper right')
    plt.show()