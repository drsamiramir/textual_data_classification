from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from src.processing.newsgroups_data_acquisition import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import unittest
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def tfidf_transformer(dataset_train, dataset_test):
    tfidf_vectorizer = TfidfVectorizer(binary=True, max_features=100000, stop_words='english')
    X_train = tfidf_vectorizer.fit_transform(dataset_train.data).astype('float32')
    X_test = tfidf_vectorizer.transform(dataset_test.data).astype('float32')
    return X_train, X_test

# texts_train = dataset_train.data
# labels_train = dataset_train.target
# texts_test = dataset_test.data
# labels_test = dataset_test.target
# print("TF-IDF on text data ... ")
# tfidf = TfidfVectorizer(binary=True, max_features=100000, stop_words='english')
# X_train = tfidf.fit_transform(texts_train).astype('float32')
# X_test = tfidf.transform(texts_test).astype('float32')
# lb = LabelEncoder()
# y = lb.fit_transform(labels_train)
# y_train = tf.keras.utils.to_categorical(y, num_classes=None, dtype="float32")
