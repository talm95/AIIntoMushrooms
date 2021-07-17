import time

import numpy as np

from clustering import cluster
from data_prepare import prepare_regular_mushroom_data, reduce_features
from plot_silhouette import plot_silhouette_graph
from machine_learning import classify
from compare_methods import plot_confusion_matrices
from sklearn.preprocessing import OneHotEncoder

start = time.time()

should_reduce_features = True
number_of_features = 3

# data preparing
full_mushrooms_data_x, full_mushrooms_data_y = prepare_regular_mushroom_data()
x_enc = OneHotEncoder()
x_enc.fit(full_mushrooms_data_x)
one_hot_encoded_full_mushrooms_data_x = x_enc.transform(full_mushrooms_data_x).toarray()
if should_reduce_features:
    one_hot_encoded_full_mushrooms_data_x = reduce_features(one_hot_encoded_full_mushrooms_data_x, number_of_features)

y_enc = OneHotEncoder()
y_enc.fit(np.array(full_mushrooms_data_y).reshape(-1, 1))
one_hot_encoded_full_mushrooms_data_y = y_enc.transform(np.array(full_mushrooms_data_y).reshape(-1, 1)).toarray()

# Clustering
for clusters_num in [7, 8, 9]:
    k_means_labeled, spectral_clustering_labeled, agglomerative_clustering_labeled = cluster(
        one_hot_encoded_full_mushrooms_data_x, clusters_num)
    plot_silhouette_graph(one_hot_encoded_full_mushrooms_data_x, k_means_labeled, spectral_clustering_labeled,
                          agglomerative_clustering_labeled, clusters_num)

# Machine Learning
labeled_data_test, labeled_data_test_encoded, random_forest_predicted, decision_tree_predicted, network_predicted = \
    classify(full_mushrooms_data_x, full_mushrooms_data_y, one_hot_encoded_full_mushrooms_data_x,
             one_hot_encoded_full_mushrooms_data_y, number_of_features)

plot_confusion_matrices(labeled_data_test_encoded, random_forest_predicted, decision_tree_predicted,
                        network_predicted, y_enc.categories_)

end = time.time()
print(end - start)
