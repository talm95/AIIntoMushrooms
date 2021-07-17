from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


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


def one_hot_encoder(data):
    enc = OneHotEncoder()
    enc.fit(data)
    return enc.transform(data).toarray()


def split_data(mushroom_data_x, mushroom_data_y):
    x_train, x_test, y_train, y_test = train_test_split(mushroom_data_x, mushroom_data_y)
    return x_train, x_test, y_train, y_test


def get_categories():
    return categories


def get_labels():
    return labels


def reduce_features(data_x, number_of_features):
    pca = PCA(n_components=number_of_features)
    data_x = pca.fit_transform(data_x)
    return data_x
