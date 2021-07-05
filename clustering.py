from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def custom_distance(mushroom1, mushroom2):
    distance_vector = []
    distance_vector = np.multiply(np.equal(mushroom1, mushroom2), 1) - 1
    # for category_num in range(np.size(mushroom1)):
    #     if mushroom1[category_num] == mushroom2[category_num]:
    #         distance_vector.append(0)
    #     else:
    #         distance_vector.append(1)
    # distance_vector = np.array(distance_vector)
    return np.linalg.norm(distance_vector)


def custom_distance_matrix(x, distance_func):
    number_of_values = np.shape(x)[0]
    distance_matrix = np.zeros(shape=(number_of_values, number_of_values))
    for row1 in range(number_of_values):
        if row1 % 300 == 0:
            print(row1)
        for row2 in range(row1 + 1, number_of_values):
            distance_matrix[row1, row2] = distance_func(np.array(x[row1, :], dtype=object), np.array(x[row2, :],
                                                                                                     dtype=object))
            distance_matrix[row2, row1] = distance_matrix[row1, row2]
    return distance_matrix


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
    # data_x = np.array(data_x[0:1000, :], dtype=object)
    # x = custom_distance_matrix(data_x, custom_distance)
    enc = OneHotEncoder()
    enc.fit(data_x)
    x = enc.transform(data_x).toarray()
    k_means_clusters = k_means(x)
    spectral_clusters = spectral_clustering(x)
    agg_clusters = agglomerative_clustering(x)

    return k_means_clusters, spectral_clusters, agg_clusters
