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
from src.preparation.feature_extraction import *


def process_training(dataset):
    """perform training process"""
    dataset_train, dataset_test = download_newsgroups_dataset()
    "to change the feature we must update the following line"
    X_train, X_test = tfidf_transformer(dataset_train, dataset_test)

    labels_train = dataset_train.target
    labels_test = dataset_test.target
    lb_encoder = LabelEncoder()
    y = lb_encoder.fit_transform(labels_train)
    y_train = tf.keras.utils.to_categorical(y, num_classes=None, dtype="float32")
    train_DL_model(X_train, y_train, X_test, y_test)

    ##Cross validation !!!
    ## data splitting !!!


def train_DL_model(X_train, y_train, X_test, y_test):
    # Model Training
    print("Create model ... ")
    estimator = KerasClassifier(build_fn=build_MLP_architecture(), epochs=10, batch_size=140)
    estimator.fit(X_train, y_train)

    # Predictions
    print("Predict on test data ... ")
    y_predict = estimator.predict(X_test)

    diff = np.sum(y_predict == y_test) / len(y_predict)

    print(len(y_predict))
    print(len(y_test))
    print(diff)


def build_MLP_architecture(self):
    model = Sequential()
    model.add(Dense(256, input_dim=100000, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dropout(0.3))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
