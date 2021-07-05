import numpy as np


categories = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
              'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
              'ring-type', 'spore-print-color', 'population', 'habitat']


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
    return np.array(mushrooms_data_x), np.array(mushrooms_data_y)


# preparing file with whole data
def prepare_regular_mushroom_data():
    regular_mushroom_data_x, regular_mushroom_data_y = prepare_data("Data/mushrooms_data.txt")
    return regular_mushroom_data_x, regular_mushroom_data_y


# preparing file with missing data
def prepare_mushroom_data_missing():
    mushroom_data_missing_x, mushroom_data_missing_y = prepare_data("Data/mushrooms_data_missing.txt")
    return mushroom_data_missing_x, mushroom_data_missing_y


def get_categories():
    return categories
