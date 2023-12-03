from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np

def word_cluster_analysis(model, num_clusters):
    word_vectors = model.wv
    vectors = word_vectors.vectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(vectors)
    word_labels = list(word_vectors.vocab.keys())
    clusters = kmeans.labels_
    return dict(zip(word_labels, clusters))
