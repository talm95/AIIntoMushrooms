from clustering import cluster
from machine_learning import classify
from data_prepare import prepare_regular_mushroom_data, prepare_mushroom_data_missing, one_hot_encoder
from compare_methods import classifiers_confusion_matrix, cluster_confusion_matrix
import numpy as np
import time

start = time.time()

# data preparing
full_mushrooms_data_x, full_mushrooms_data_y = prepare_regular_mushroom_data()
one_hot_encoded_full_mushrooms_data_x = one_hot_encoder(full_mushrooms_data_x)
one_hot_encoded_full_mushrooms_data_y = one_hot_encoder(np.array(full_mushrooms_data_y).reshape(-1, 1))

# Clustering
k_means_labeled, spectral_clustering_labeled, agglomerative_clustering_labeled = cluster(
    one_hot_encoded_full_mushrooms_data_x)

k_means_conf_matrix = cluster_confusion_matrix(one_hot_encoded_full_mushrooms_data_y, k_means_labeled)
print('k_means confusion matrix')
print(k_means_conf_matrix)

spectral_clustering_conf_matrix = cluster_confusion_matrix(one_hot_encoded_full_mushrooms_data_y,
                                                           spectral_clustering_labeled)
print('spectral clustering confusion matrix')
print(spectral_clustering_conf_matrix)

agglomerative_clustering_conf_matrix = cluster_confusion_matrix(one_hot_encoded_full_mushrooms_data_y,
                                                                agglomerative_clustering_labeled)
print('agglomerative clustering confusion matrix')
print(agglomerative_clustering_conf_matrix)

# Machine Learning
labeled_data_test, labeled_data_test_encoded, random_forest_predicted, decision_tree_predicted = \
    classify(full_mushrooms_data_x, full_mushrooms_data_y, one_hot_encoded_full_mushrooms_data_x,
             one_hot_encoded_full_mushrooms_data_y)

rf_confusion_matrix = classifiers_confusion_matrix(labeled_data_test_encoded, random_forest_predicted)
print('random forest confusion matrix')
print(rf_confusion_matrix)

dt_confusion_matrix = classifiers_confusion_matrix(labeled_data_test_encoded, decision_tree_predicted)
print('decision tree confusion matrix')
print(dt_confusion_matrix)

end = time.time()
print(end - start)
