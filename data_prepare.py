import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


categories = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
              'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
              'ring-type', 'spore-print-color', 'population', 'habitat']

labels = {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 'n': 'none',
          'p': 'pungent', 's': 'spicy'}


# generic function for preparing mushroom data based on file
def prepare_data(file_name):
    mushrooms_data_file = open(file_name, "r")
    mushrooms_data_lines = mushrooms_data_file.readlines()
    mushrooms_data_x = []
    mushrooms_data_y = []
    for line in mushrooms_data_lines:
        line = line.replace('\n', '')
        line_values = line.split(',')
        x_line = []
        for category_num, category in enumerate(categories):
            if category == 'odor':
                mushrooms_data_y.append(line_values[category_num])
            else:
                x_line.append(line_values[category_num])
        mushrooms_data_x.append(x_line)

    return mushrooms_data_x, mushrooms_data_y


# preparing file with whole data
def prepare_regular_mushroom_data():
    return prepare_data("Data/mushrooms_data.txt")


# preparing file with missing data
def prepare_mushroom_data_missing():
    return prepare_data("Data/mushrooms_data_missing.txt")


def split_unlabeled_data(missing_data_x, missing_data_y):
    missing_label_indices = np.where(missing_data_y == '-')
    missing_labels_y = missing_data_y[missing_label_indices]
    missing_labels_x = missing_data_x[missing_label_indices]
    labeled_y = np.delete(missing_data_y, missing_label_indices)
    labeled_x = np.delete(missing_data_x, missing_label_indices, axis=0)

    return labeled_x, missing_labels_x, labeled_y, missing_labels_y


def get_categories():
    return categories


def reduce_features(data_x, number_of_features):
    pca = PCA(n_components=number_of_features)
    data_x = pca.fit_transform(data_x)
    return data_x


def plot_explained_variance(data_x):
    pca = PCA()
    pca.fit(data_x)
    variance_explained = pca.explained_variance_ratio_.cumsum()
    plt.plot(range(1, pca.n_components_ + 1)[::3], variance_explained[::3], '*')
    plt.title('Variance explained respective to number of features')
    plt.xlabel('# of features')
    plt.ylabel('variance explained')
    plt.show()
