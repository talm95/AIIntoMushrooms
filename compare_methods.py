from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
import matplotlib.pyplot as plt
from data_prepare import labels
import numpy as np
import math


def plot_clusters_confusion_matrices(real_labels, k_means_clusters, spectral_clusters, agglomerative_clusters,
                                     clusters_num):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    k_means_recall = plot_cluster_confusion_matrix(ax1, real_labels, k_means_clusters)
    ax1.set_title('K means Clusters Confusion Matrix\nrecall score = %.4f' % k_means_recall)

    spectral_recall = plot_cluster_confusion_matrix(ax2, real_labels, spectral_clusters)
    ax2.set_title('Spectral Clusters Confusion Matrix\nrecall score = %.4f' % spectral_recall)

    agglomerative_recall = plot_cluster_confusion_matrix(ax3, real_labels, agglomerative_clusters)
    ax3.set_title('Agglomerative Clusters Confusion Matrix\nrecall score = %.4f' % agglomerative_recall)

    fig.suptitle('Confusion Matrices for ' + str(clusters_num) + ' clusters')
    plt.show()

    return [k_means_recall, spectral_recall, agglomerative_recall]


def plot_cluster_confusion_matrix(ax, real_labels, clusters):
    matrix = confusion_matrix(real_labels.argmax(axis=1), clusters)
    new_matrix, mapping = rearrange_matrix(matrix)
    rearranged_clusters = rearrange_clusters(clusters, mapping)
    matrix_to_display = ConfusionMatrixDisplay(new_matrix)
    matrix_to_display.plot(ax=ax, cmap=plt.cm.get_cmap('Blues'), values_format='.0f', colorbar=False)
    recall = recall_score(real_labels.argmax(axis=1), rearranged_clusters, average='micro')
    return recall


def plot_recall_score_per_clusters_num(clusters_nums, recall_scores):
    plt.plot(clusters_nums, recall_scores[:, 0], 'b', label="k means")
    plt.plot(clusters_nums, recall_scores[:, 1], 'r', label="spectral clustering")
    plt.plot(clusters_nums, recall_scores[:, 2], 'k', label="agglomertaive clustering")
    plt.legend(loc='upper left')
    plt.xlabel('clusters number')
    plt.ylabel('recall score')
    plt.title('recall scores by clusters number')
    plt.show()


def plot_confusion_matrices(real_y_encoded, random_forest_predicted_y, decision_tree_predicted_y, network_predicted_y,
                            encoded_labels):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    plot_sub_confusion_matrix(real_y_encoded.argmax(axis=1), random_forest_predicted_y.argmax(axis=1), ax1,
                              encoded_labels[0])
    random_forest_recall = recall_score(real_y_encoded, random_forest_predicted_y, average='micro')
    ax1.set_title('Random Forest Confusion Matrix\nrecall = ' + str(random_forest_recall))

    plot_sub_confusion_matrix(real_y_encoded.argmax(axis=1), decision_tree_predicted_y.argmax(axis=1), ax2,
                              encoded_labels[0])
    decision_tree_recall = recall_score(real_y_encoded, decision_tree_predicted_y, average='micro')
    ax2.set_title('Decision Tree Confusion Matrix\nrecall = ' + str(decision_tree_recall))

    plot_sub_confusion_matrix(real_y_encoded.argmax(axis=1), network_predicted_y.argmax(axis=1), ax3, encoded_labels[0])
    network_recall = recall_score(real_y_encoded.argmax(axis=1), network_predicted_y.argmax(axis=1), average='micro')
    ax3.set_title('Neural Network Confusion Matrix\nrecall = ' + str(network_recall))

    plt.show()


def plot_sub_confusion_matrix(real_y_encoded, predicted_y, ax, real_labels):
    matrix = confusion_matrix(real_y_encoded, predicted_y, labels=list(range(len(labels))))
    matrix_to_display = ConfusionMatrixDisplay(matrix, real_labels)
    matrix_to_display.plot(ax=ax, cmap=plt.cm.get_cmap('Blues'), colorbar=False)


def plot_loss_graph(epochs, losses):
    epochs_axis = np.array(range(math.floor(epochs/5))) * 5
    plt.plot(epochs_axis, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Vs Epoch')
    plt.show()


def plot_clusters(pcs, y, clusters_num, predicted=True):
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'orange', 'brown', 'cyan', 'olive', 'purple']
    pcs_colors = []
    for yi in y:
        pcs_colors.append(colors[yi])
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=pcs_colors)
    if predicted:
        ax.set_title("Predicted clusters. Clusters number - " + str(clusters_num))
    else:
        ax.set_title("Actual clusters. Clusters number - " + str(clusters_num))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_ylabel('PC3')
    plt.show()


def rearrange_matrix(matrix):
    new_matrix = np.zeros(shape=matrix.shape)
    altering_matrix = np.copy(matrix)
    mapping = np.array(range(matrix.shape[0]))
    for col in range(matrix.shape[0]):
        largest_indices = np.where(altering_matrix == np.amax(altering_matrix))
        new_matrix[:, largest_indices[0][0]] = matrix[:, largest_indices[1][0]]
        altering_matrix[:, largest_indices[1][0]] -= 5000
        altering_matrix[largest_indices[0][0], :] -= 5000
        mapping[largest_indices[1][0]] = largest_indices[0][0]
    return new_matrix, mapping


def rearrange_clusters(clusters, mapping):
    new_clusters = [mapping[cluster] for cluster in clusters]
    return new_clusters
