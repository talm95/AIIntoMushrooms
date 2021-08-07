import numpy as np

from data_prepare import prepare_mushroom_data_missing, split_unlabeled_data
from neural_network import Network
from supervised_learning import neural_network_classifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


number_of_features = 77
number_of_outputs = 5

missing_data_x, missing_data_y = prepare_mushroom_data_missing()
missing_data_y = np.array(missing_data_y)

categories_to_drop_x = ['-'] * 21
missing_data_x.append(['-'] * 21)

x_enc = OneHotEncoder(drop=categories_to_drop_x)
x_enc.fit(missing_data_x)
missing_data_x = missing_data_x[:-1]
missing_data_x_encoded = x_enc.transform(missing_data_x).toarray()

labeled_x, unlabeled_x, labeled_y_not_enc, unlabeled_y_not_enc = \
    split_unlabeled_data(missing_data_x_encoded, missing_data_y)

y_enc = OneHotEncoder()
y_enc.fit(labeled_y_not_enc.reshape(-1, 1))
labeled_y = y_enc.transform(labeled_y_not_enc.reshape(-1, 1)).toarray()

network = Network(number_of_features, number_of_outputs)
nn_predicted = neural_network_classifier(network, labeled_x, unlabeled_x, labeled_y)

nn_predicted_numpy = nn_predicted.detach().numpy()

full_data_train = np.append(labeled_x, labeled_y.argmax(axis=1).reshape(-1, 1), axis=1)

full_data_test = np.append(unlabeled_x, nn_predicted.argmax(axis=1).reshape(-1, 1), axis=1)

clf = LocalOutlierFactor(novelty=True, n_neighbors=100)
clf.fit(full_data_train)

outliers = clf.predict(full_data_test)

num_of_outliers = np.count_nonzero(outliers == -1)
num_of_inliers = len(outliers) - num_of_outliers

fig, ax = plt.subplots(1, 1)
x_axis = ['Inliers', 'Outliers']
y_axis = [num_of_inliers, num_of_outliers]
ax.bar(x_axis, y_axis)
ax.set_title('Number of inliers and outliers')
ax.set_xlabel('Label')
ax.set_ylabel('Count')

plt.show()
