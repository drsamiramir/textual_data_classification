import unittest
from tensorflow.python.keras.datasets import imdb
from src.modeling.lstm_cnn_classifier import *


class LSTM_CNN_ClassifierTest(unittest.TestCase):

    def test_lstm_cnn_classifier(self):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
        lstm_cnn_classifier = LSTM_CNN_Classifier(X_train, y_train, X_test, y_test)
        lstm_cnn_classifier.train_dl_model()
