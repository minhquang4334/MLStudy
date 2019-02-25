
# -*- encoding: utf-8 -*-

import numpy as np

import re
import io
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pyvi import ViTokenizer, ViPosTagger
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


documents = []


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
    result = " ".join([word for word in temp if is_not_in_stop_words(word, stop_words)])
    return result.lower()


def tokenizer(str):
    return ViTokenizer.tokenize(str)


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

    # print arr_tf_idf
    return arr_tf_idf


def sklearn_tf_idf(docs):
    print len(docs)
    tfidfconverter = TfidfVectorizer()
    x = tfidfconverter.fit_transform(docs).toarray()

    return x


def logistic_regression(X_train, y_train, X_test, y_test):
    lr_clf = LogisticRegression(penalty='l2', dual=False, tol=1e-5, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None)
    lr_clf.fit(X_train, y_train)
    predicted = lr_clf.predict(X_test)
    np.mean(predicted == y_test)
    return predicted


dataset = get_datasets_localdata("../dataset1/sub_train/")

y = dataset.target

f = io.open('./stopwords.txt', 'r')
array_stop_words = f.read().splitlines()

for index in range(0, len(dataset.data)):
    token = tokenizer(dataset.data[index])
    documents.append(vietnamese_pre_processing(token, array_stop_words))
    # print type(dataset.data[index])

X = sklearn_tf_idf(documents)

# X = manual_compute_tf_idf(documents)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

predicted = logistic_regression(X_train, y_train, X_test, y_test)
print predicted, y_test
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))
print(accuracy_score(y_test, predicted))


