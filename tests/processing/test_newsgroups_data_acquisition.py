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


class NewsgroupAcquistitionTest(unittest.TestCase):

    def test_download_dataset(self):
        dataset_train, dataset_test = download_dataset()
        categories = dataset_train.target_names
        labels_train = dataset_train.target
        texts_train = dataset_train.data
        texts_test = dataset_test.data
        labels_test = dataset_test.target
        # pprint((texts)[1])

        # Feature Engineering
        print("TF-IDF on text data ... ")
        tfidf = TfidfVectorizer(binary=True)
        X_train = tfidf.fit_transform(texts_train).astype('float32')
        X_test = tfidf.transform(texts_test).astype('float32')

        print("Label Encode the Target Variable ... ")
        lb = LabelEncoder()
        y = lb.fit_transform(labels_train)
        print(y)
        y_train = tf.keras.utils.to_categorical(y, num_classes=None, dtype="float32")
        # Model Training
        print("Create model ... ")

        estimator = KerasClassifier(build_fn=self.build_model, epochs=10, batch_size=100)
        estimator.fit(X_train, y_train)

    # Predictions
        print ("Predict on test data ... ")
        y_predict = estimator.predict(X_test)
        y_test = lb.transform(labels_test)
        diff = np.sum(y_predict == y_test)/len(y_predict)



        print(len(y_predict))
        print(len(y_test))
        print((diff))








    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=130107, activation='relu'))
        model.add(Dropout(0.3))
        # model.add(Dense(200, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(160, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(20, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model


