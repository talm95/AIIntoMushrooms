import time

import numpy as np

from clustering import cluster
from data_prepare import prepare_regular_mushroom_data, reduce_features, plot_explained_variance
from plot_silhouette import plot_silhouette_graph, plot_silhouette_score_per_cluster_num
from machine_learning import classify
from compare_methods import plot_confusion_matrices, plot_clusters_confusion_matrices,\
    plot_recall_score_per_clusters_num
from sklearn.preprocessing import OneHotEncoder

start = time.time()

should_reduce_features = True
number_of_features = 43
if not should_reduce_features:
    number_of_features = 101

# data preparing
full_mushrooms_data_x, full_mushrooms_data_y = prepare_regular_mushroom_data()
x_enc = OneHotEncoder()
x_enc.fit(full_mushrooms_data_x)
one_hot_encoded_full_mushrooms_data_x = x_enc.transform(full_mushrooms_data_x).toarray()

# uncomment to see variance explained based on number of features
# plot_explained_variance(one_hot_encoded_full_mushrooms_data_x)

if should_reduce_features:
    one_hot_encoded_full_mushrooms_data_x = reduce_features(one_hot_encoded_full_mushrooms_data_x, number_of_features)

y_enc = OneHotEncoder()
y_enc.fit(np.array(full_mushrooms_data_y).reshape(-1, 1))
one_hot_encoded_full_mushrooms_data_y = y_enc.transform(np.array(full_mushrooms_data_y).reshape(-1, 1)).toarray()

# Clustering
clusters_nums = [5, 6, 7, 8, 9]
silhouette_scores_per_clusters_num = []
recall_scores_per_clusters_num = []
for clusters_num in clusters_nums:
    k_means_labeled, spectral_clustering_labeled, agglomerative_clustering_labeled = cluster(
        one_hot_encoded_full_mushrooms_data_x, clusters_num)
    silhouette_scores = plot_silhouette_graph(one_hot_encoded_full_mushrooms_data_x, k_means_labeled,
                                              spectral_clustering_labeled, agglomerative_clustering_labeled,
                                              clusters_num)
    silhouette_scores_per_clusters_num.append(silhouette_scores)
    # if should_reduce_features and number_of_features == 3:
    #     plot_clusters(one_hot_encoded_full_mushrooms_data_x, agglomerative_clustering_labeled, clusters_num)
    #     plot_clusters(one_hot_encoded_full_mushrooms_data_x, one_hot_encoded_full_mushrooms_data_y.argmax(axis=1),
    #                   clusters_num, predicted=False)
    recalls = plot_clusters_confusion_matrices(one_hot_encoded_full_mushrooms_data_y, k_means_labeled,
                                               spectral_clustering_labeled, agglomerative_clustering_labeled,
                                               clusters_num)

    recall_scores_per_clusters_num.append(recalls)
plot_silhouette_score_per_cluster_num(clusters_nums, np.array(silhouette_scores_per_clusters_num))
plot_recall_score_per_clusters_num(clusters_nums, np.array(recall_scores_per_clusters_num))


# Machine Learning
labeled_data_test, labeled_data_test_encoded, random_forest_predicted, decision_tree_predicted, network_predicted = \
    classify(full_mushrooms_data_x, full_mushrooms_data_y, one_hot_encoded_full_mushrooms_data_x,
             one_hot_encoded_full_mushrooms_data_y, number_of_features)

plot_confusion_matrices(labeled_data_test_encoded, random_forest_predicted, decision_tree_predicted,
                        network_predicted, y_enc.categories_)

end = time.time()
print(end - start)
