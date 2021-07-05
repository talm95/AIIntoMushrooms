from clustering import cluster
from data_prepare import *
from compare_methods import cluster_confusion_matrix
import time

start = time.time()

full_mushrooms_data_x, full_mushrooms_data_y = prepare_regular_mushroom_data()
k_means_labeled, spectral_clustering_labeled, agglomerative_clustering_labeled = cluster(full_mushrooms_data_x)

k_means_conf_matrix = cluster_confusion_matrix(k_means_labeled, full_mushrooms_data_y)
print('k_means confusion matrix')
print(k_means_conf_matrix)

spectral_clustering_conf_matrix = cluster_confusion_matrix(spectral_clustering_labeled, full_mushrooms_data_y)
print('spectral_clustering_confusion_matrix')
print(spectral_clustering_conf_matrix)

agglomerative_clustering_conf_matrix = cluster_confusion_matrix(agglomerative_clustering_labeled, full_mushrooms_data_y)
print('agglomerative_clustering_confusion_matrix')
print(agglomerative_clustering_conf_matrix)

end = time.time()
print(end - start)
