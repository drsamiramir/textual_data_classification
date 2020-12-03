import unittest
from src.modeling.mlp_classifier import *
from src.processing.newsgroups_data_acquisition import *
from src.preparation.feature_extraction import *
from src.modeling.model_evaluation import *

class DLClassificationTest(unittest.TestCase):
    def test_mlp_classifier(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        X_train, X_test = tfidf_transformer(dataset_train, dataset_test)
        # X_train, X_test = count_vectorizer(dataset_train, dataset_test)
        y_train = dataset_train.target
        y_test = dataset_test.target
        mlp_classifier = MLP_Classifier(X_train, y_train, X_test, y_test)
        model = mlp_classifier.perform_training()
        evaluate_model(model, X_test, y_test)
