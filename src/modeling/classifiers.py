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


def create_MLP_model(X_train, y_train, X_test, y_test):
    # Model Training
    print("Create model ... ")

    lb_encoder = LabelEncoder()
    estimator = KerasClassifier(build_fn=build_model, epochs=10, batch_size=140)
    estimator.fit(X_train, y_train)

    # Predictions
    print("Predict on test data ... ")
    y_predict = estimator.predict(X_test)
    y_test = lb_encoder.transform(y_test)
    diff = np.sum(y_predict == y_test) / len(y_predict)

    print(len(y_predict))
    print(len(y_test))
    print(diff)


def build_model(self):
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
