from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def random_forest_classifier(data_x_train, data_x_test, data_y_train):
    rf_classifier = RandomForestClassifier(criterion='entropy')
    rf_classifier.fit(data_x_train, data_y_train)
    return rf_classifier.predict(data_x_test)


def decision_tree_classifier(data_x_train, data_x_test, data_y_train):
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(data_x_train, data_y_train)
    return dt_classifier.predict(data_x_test)


def classify(data_x, data_y, data_x_encoded, data_y_encoded):
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y)
    data_x_train_encoded, data_x_test_encoded, data_y_train_encoded, data_y_test_encoded = train_test_split(
        data_x_encoded, data_y_encoded)
    rf_predicted_y = random_forest_classifier(data_x_train_encoded, data_x_test_encoded, data_y_train_encoded)
    lr_predicted_y = decision_tree_classifier(data_x_train_encoded, data_x_test_encoded, data_y_train_encoded)
    return data_y_test, data_y_test_encoded, rf_predicted_y, lr_predicted_y
