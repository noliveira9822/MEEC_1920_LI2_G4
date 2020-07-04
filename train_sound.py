from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sound_utils import load_commands_data, load_groups_data
import os
import pickle


def init_model_params():
    model_params = {
        'alpha': 0.01,
        # 'batch_size': 254,
        'epsilon': 1e-8,
        'hidden_layer_sizes': (600, 80),
        'learning_rate': 'adaptive',
        'max_iter': 500,
    }
    return model_params


def train_commands_model(test_size):
    model_params = init_model_params()
    model_commands = MLPClassifier(**model_params)
    X_train, X_test, y_train, y_test = load_commands_data(test_size=test_size)
    print("Training model....")
    model_commands.fit(X_train, y_train)
    y_pred = model_commands.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    if not os.path.isdir("sound_models"):
        os.mkdir("sound_models")

    pickle.dump(model_commands, open("sound_models/mlp_classifier_commands_0407.model", "wb"))


def train_groups_model(test_size):
    model_params = init_model_params()
    model_groups = MLPClassifier(**model_params)
    X_train, X_test, y_train, y_test = load_groups_data(test_size=test_size)
    print("Training model....")
    model_groups.fit(X_train, y_train)
    y_pred = model_groups.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    if not os.path.isdir("sound_models"):
        os.mkdir("sound_models")

    pickle.dump(model_groups, open("sound_models/mlp_classifier_groups_0407.model", "wb"))
