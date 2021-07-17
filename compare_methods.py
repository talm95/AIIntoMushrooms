from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
import matplotlib.pyplot as plt
from data_prepare import labels
import numpy as np
import math


def cluster_confusion_matrix(real_labels, clusters):
    return confusion_matrix(real_labels.argmax(axis=1), clusters)


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
    matrix_to_display.plot(ax=ax, cmap=plt.cm.get_cmap('Blues'))


def plot_loss_graph(epochs, losses):
    epochs_axis = np.array(range(math.floor(epochs/5))) * 5
    plt.plot(epochs_axis, losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
