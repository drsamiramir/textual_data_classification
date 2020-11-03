import unittest
from src.modeling.deep_learning_classifiers import *
from src.processing.newsgroups_data_acquisition import *
from src.preparation.feature_extraction import *
from src.modeling.model_evaluation import *
import numpy as np


class DLClassificationTest(unittest.TestCase):
    # @unittest.skip("Test ignored (used only experiment)")
    def test_dl_classifier(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        X_train, X_test = tfidf_transformer(dataset_train, dataset_test)
        #X_train, X_test = count_vectorizer(dataset_train, dataset_test)
        y_train = dataset_train.target
        y_test = dataset_test.target
        dl_classifier = DeepLearningClassifier(X_train.shape[1], len(np.unique(y_train)), X_train, y_train)
        model = dl_classifier.process_training()
        evaluate_dl_model(model, X_test, y_test)
