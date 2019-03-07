import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pyvi import ViTokenizer, ViPosTagger
import re
from sklearn.datasets import load_files
import io
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                       encoding='utf-16', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


model = joblib.load('text_classifier.sav')

test_data = get_datasets_localdata("../dataset1/test_sub_train/")
X_test, y_test = test_data.data, test_data.target

# print model.predict(X_test)
y_pred2 = model.score(X_test, y_test)
print "y_pred: ", y_pred2

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))