import unittest
from src.modeling.deep_learning_classifiers import *
from src.processing.newsgroups_data_acquisition import *


class ClassificationTest(unittest.TestCase):
    #@unittest.skip("Test ignored (used only experiment)")
    def test_dl_classifier(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        dl_classifier = DeepLearningClassifier
        dl_classifier.process_training(dl_classifier, dataset_train,dataset_test)



