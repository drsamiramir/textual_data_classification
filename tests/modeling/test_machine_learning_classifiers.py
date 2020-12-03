import unittest
from src.modeling.cnn_classifier import *
from src.modeling.machine_learning_classifier import *
from src.processing.newsgroups_data_acquisition import *


class MLClassificationTest(unittest.TestCase):
    #@unittest.skip("Test ignored (used only experiment)")
    def test_ml_classifier(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        ml_classifier = MachineLearningClassifier
        ml_classifier.process_training(ml_classifier, dataset_train,dataset_test)





