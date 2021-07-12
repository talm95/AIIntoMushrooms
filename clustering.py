from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


def k_means(x, clusters_num):
    k_means_alg = KMeans(n_clusters=clusters_num)
    return k_means_alg.fit_predict(x)


def spectral_clustering(x, clusters_num):
    s_clustering = SpectralClustering(n_clusters=clusters_num)
    return s_clustering.fit_predict(x)


def agglomerative_clustering(x, clusters_num):
    agg_clustering = AgglomerativeClustering(n_clusters=clusters_num)
    return agg_clustering.fit_predict(x)


def cluster(data_x, clusters_num):
    k_means_clusters = k_means(data_x, clusters_num)
    spectral_clusters = spectral_clustering(data_x, clusters_num)
    agg_clusters = agglomerative_clustering(data_x, clusters_num)

    return k_means_clusters, spectral_clusters, agg_clusters
