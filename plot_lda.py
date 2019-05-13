import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def plot_lda(docs_repr):
    
    topics = docs_repr.argmax(axis=1) + 1
    unique_topics = list(set(topics))
    colors = sns.color_palette("hls", 7)
    
    X_embedded = TSNE(n_components=2).fit_transform(docs_repr)
    data = pd.DataFrame(np.hstack((X_embedded, topics[:,None])))
    data[2] = data[2].astype(int)
    
    f, ax = plt.subplots()
    data = data[data[2].isin([1, 2, 16, 10, 11, 17, 20])]
    for i in range(len([1, 2, 16, 10, 11, 17, 20])):
        d = data[data[2] == [1, 2, 16, 10, 11, 17, 20][i]]
        if len(d) > 0:
            ax.scatter(d[0], d[1], color=colors[i], label=[1, 2, 16, 10, 11, 17, 20][i])
    ax.legend(loc='upper left')
    return f

def corr(docs_repr):
    
    data = pd.DataFrame(docs_repr)
    data.columns = (np.array(data.columns)+1).tolist()
    corr = data.corr()
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0, square = True, linewidth = .2)
    return f