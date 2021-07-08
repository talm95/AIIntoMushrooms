from sklearn.metrics import confusion_matrix


def cluster_confusion_matrix(real_labels, clusters):
    return confusion_matrix(real_labels.argmax(axis=1), clusters)


def classifiers_confusion_matrix(real_y, predicted_y):
    return confusion_matrix(real_y.argmax(axis=1), predicted_y.argmax(axis=1))
