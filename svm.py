
# -*- encoding: utf-8 -*-

import numpy as np

import re
import io
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from pyvi import ViTokenizer, ViPosTagger
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

from sklearn import svm
import gc
import time
import pickle
start_time = time.time()



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


def translate_non_alphanumerics(to_translate, translate_to=u'_'):
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^`{|}~'
    translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
    return to_translate.translate(translate_table)


def is_not_in_stop_words(word, stop_words):
    return word not in stop_words


def vietnamese_pre_processing(text, stop_words):
    text = translate_non_alphanumerics(text, u" ")
    doc = re.sub(r'\d+', '', text)
    tmp = re.sub(r'\s+', ' ', doc, flags=re.I)
    temp = tmp.split()
    generator = (word for word in temp if is_not_in_stop_words(word, stop_words))
    result = " ".join(word for word in generator)
    return result.lower()


def tokenizer(str):
    return (ViTokenizer.tokenize(str))


def total_words_in_document(document):
    return len(document.split(' '))


def svm_classification(X_train, y_train, X_test, y_test):
    svm_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svm_clf.fit(X_train, y_train)
    predicted = svm_clf.predict(X_test)
    np.mean(predicted == y_test)
    return predicted, svm_clf


def get_data(data, array_stop_words):
    documents = []
    for index in range(0, len(data)):
        token = tokenizer(data[index])
        documents.append(vietnamese_pre_processing(token, array_stop_words))
    return documents


dataset = get_datasets_localdata("../dataset1/sub_train/")
test_data = get_datasets_localdata("../dataset1/test_sub_train/")

f = io.open('./stopwords.txt', 'r')
array_stop_words = f.read().splitlines()

X_train, y_train = dataset.data, dataset.target
X_test, y_test = test_data.data, test_data.target

X_train = get_data(X_train, array_stop_words)
X_test = get_data(X_test, array_stop_words)

tfidf = TfidfVectorizer(norm='l2')
X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

predicted, classifier = svm_classification(X_train, y_train, X_test, y_test)

# predicted = logistic_regression(X_train, y_train, X_test, y_test)

print predicted, y_test
print(classification_report(y_test, predicted))
print(accuracy_score(y_test, predicted))


elapsed_time = time.time() - start_time

print "Total_Time for Excute: ", elapsed_time

