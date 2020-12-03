import unittest
from src.modeling.cnn_classifier import *
from src.processing.newsgroups_data_acquisition import *


class CNN_ClassifierTest(unittest.TestCase):

    def test_cnn_classifier(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        X_train = dataset_train.data
        X_test = dataset_test.data
        y_train = dataset_train.target
        y_test = dataset_test.target
        fc_cnn_classifier = FC_CNN_Classifier(X_train, y_train, X_test, y_test)
        fc_cnn_classifier.perform_training()
