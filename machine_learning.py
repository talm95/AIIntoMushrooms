import torch.nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch import optim

from neural_network import Network


def random_forest_classifier(data_x_train, data_x_test, data_y_train):
    rf_classifier = RandomForestClassifier(criterion='entropy')
    rf_classifier.fit(data_x_train, data_y_train)
    return rf_classifier.predict(data_x_test)


def decision_tree_classifier(data_x_train, data_x_test, data_y_train):
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(data_x_train, data_y_train)
    return dt_classifier.predict(data_x_test)


def neural_network_classifier(network, data_x_train, data_x_test, data_y_train):
    network = network.double()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.01)
    epochs = 150
    x = torch.from_numpy(data_x_train).split(16)
    y = torch.from_numpy(data_y_train.argmax(axis=1)).split(16)
    for e in range(epochs):
        for x_tensor, y_tensor in zip(x, y):

            optimizer.zero_grad()

            output = network(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

    return network(torch.from_numpy(data_x_test))


def classify(data_x, data_y, data_x_encoded, data_y_encoded):
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y)
    data_x_train_encoded, data_x_test_encoded, data_y_train_encoded, data_y_test_encoded = train_test_split(
        data_x_encoded, data_y_encoded)
    network = Network()

    rf_predicted_y = random_forest_classifier(data_x_train_encoded, data_x_test_encoded, data_y_train_encoded)
    lr_predicted_y = decision_tree_classifier(data_x_train_encoded, data_x_test_encoded, data_y_train_encoded)
    nn_predicted_y = neural_network_classifier(network, data_x_train_encoded, data_x_test_encoded, data_y_train_encoded)

    return data_y_test, data_y_test_encoded, rf_predicted_y, lr_predicted_y, nn_predicted_y
