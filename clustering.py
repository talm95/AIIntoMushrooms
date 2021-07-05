from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import OneHotEncoder


def k_means(x):
    k_means_alg = KMeans(n_clusters=9)
    return k_means_alg.fit_predict(x)


def spectral_clustering(x):
    s_clustering = SpectralClustering(n_clusters=9)
    return s_clustering.fit_predict(x)


def agglomerative_clustering(x):
    agg_clustering = AgglomerativeClustering(n_clusters=9)
    return agg_clustering.fit_predict(x)


def cluster(data_x):
    enc = OneHotEncoder()
    enc.fit(data_x)
    x = enc.transform(data_x).toarray()
    k_means_clusters = k_means(x)
    spectral_clusters = spectral_clustering(x)
    agg_clusters = agglomerative_clustering(x)

    return k_means_clusters, spectral_clusters, agg_clusters
