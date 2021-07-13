from sklearn.metrics import confusion_matrix, silhouette_score, ConfusionMatrixDisplay, plot_confusion_matrix
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
    ax1.set_title('Random Forest Confusion Matrix')

    plot_sub_confusion_matrix(real_y_encoded.argmax(axis=1), decision_tree_predicted_y.argmax(axis=1), ax2,
                              encoded_labels[0])
    ax2.set_title('Decision Tree Confusion Matrix')

    plot_sub_confusion_matrix(real_y_encoded.argmax(axis=1), network_predicted_y.argmax(axis=1), ax3, encoded_labels[0])
    ax3.set_title('Neural Network Confusion Matrix')

    plt.show()


def plot_sub_confusion_matrix(real_y_encoded, predicted_y, ax, real_labels):
    matrix = confusion_matrix(real_y_encoded, predicted_y, labels=list(range(len(labels))), normalize='true')
    matrix_to_display = ConfusionMatrixDisplay(matrix, real_labels)
    matrix_to_display.plot(ax=ax, cmap=plt.cm.get_cmap('Blues'))


def plot_loss_graph(epochs, losses):
    epochs_axis = np.array(range(math.floor(epochs/5))) * 5
    plt.plot(epochs_axis, losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
