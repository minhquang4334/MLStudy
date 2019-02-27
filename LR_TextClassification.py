
# -*- encoding: utf-8 -*-

import numpy as np

import re
import io
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
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
    tmp = re.sub(r'\s+', ' ', text, flags=re.I)
    temp = tmp.split()
    generator = (word for word in temp if is_not_in_stop_words(word, stop_words))
    result = " ".join(word for word in generator)
    return result.lower()


def tokenizer(str):
    return (ViTokenizer.tokenize(str))


def total_words_in_document(document):
    return len(document.split(' '))


def compute_tf(word_dict, total_words):
    tf_dict = {}
    for word, count in word_dict.items():
        tf_dict[word] = count/float(total_words)
    return tf_dict


def compute_idf(doc_list):
    import math
    idf_dict = {}
    n = len(doc_list)
    # print doc_list[0].keys()

    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                if word in idf_dict:
                    idf_dict[word] = idf_dict[word] + 1
                else:
                    idf_dict[word] = 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(1 + n / float(val))

    return idf_dict


def compute_tf_idf(tf, idfs):
    tf_idf = {}
    for word, val in tf.items():
        tf_idf[word] = val * idfs[word]

    return tf_idf


def manual_compute_tf_idf(docs):
    tf = []
    corpus = []
    v_dict = {}
    for i in range(0, len(docs)):
        arr_words = docs[i].split()
        dict_words = dict.fromkeys(arr_words, 0)
        for word in arr_words:
            dict_words[word] += 1
            v_dict[word] = 0

        corpus.append(dict_words)
        tf.append(compute_tf(dict_words, len(arr_words)))

    idf = compute_idf(corpus)
    tf_idf = []
    for i in range(0, len(tf)):
        tf_idf.append(compute_tf_idf(tf[i], idf))

    arr_tf_idf = []
    for i in range(0, len(tf_idf)):
        temp = tf_idf[i]
        arr = []
        for word, val in v_dict.items():
            if word not in temp:
                arr.append(0)
            else:
                arr.append(temp[word])
        arr_tf_idf.append(arr)
    gc.collect()
    return arr_tf_idf


def logistic_regression(X_train, y_train, X_test, y_test):
    lr_clf = LogisticRegression(C=1.2,
                           max_iter=100,
                           fit_intercept=False,
                           n_jobs=3,
                           random_state=0,
                           solver='lbfgs',
                           multi_class='multinomial'
                        )
    lr_clf.fit(X_train, y_train)

    predicted = lr_clf.predict(X_test)

    np.mean(predicted == y_test)
    return predicted, lr_clf


def sgd_classification(X_train, y_train, X_test, y_test):
    sgd_clf = SGDClassifier()
    sgd_clf.fit(X_train, y_train)
    predicted = sgd_clf.predict(X_test)
    np.mean(predicted == y_test)
    return predicted, sgd_clf


def random_forest(X_train, y_train, X_test, y_test):
    rf_clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    rf_clf.fit(X_train, y_train)
    predicted = rf_clf.predict(X_test)
    np.mean(predicted == y_test)
    return predicted, rf_clf


def naive_bayes(X_train, y_train, X_test, y_test):
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    predicted = nb_clf.predict(X_test)
    np.mean(predicted == y_test)
    return predicted, nb_clf


def k_neighbors(X_train, y_train, X_test, y_test):
    kn_clf = KNeighborsClassifier(3)
    kn_clf.fit(X_train, y_train)
    predicted = kn_clf.predict(X_test)
    np.mean(predicted == y_test)
    return predicted, kn_clf


def neural_network(X_train, y_train, X_test, y_test):
    kn_clf = MLPClassifier(alpha=1)
    kn_clf.fit(X_train, y_train)
    predicted = kn_clf.predict(X_test)
    np.mean(predicted == y_test)
    return predicted, kn_clf


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

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

predicted, classifier = logistic_regression(X_train, y_train, X_test, y_test)

# predicted = logistic_regression(X_train, y_train, X_test, y_test)

print predicted, y_test
print(classification_report(y_test, predicted))
print(accuracy_score(y_test, predicted))


elapsed_time = time.time() - start_time

print "Total_Time for Excute: ", elapsed_time

with open('text_classifier.sav', 'wb') as joblibfile:
    joblib.dump(classifier, joblibfile)