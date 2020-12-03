import unittest
from src.modeling.deep_learning_classifiers import *
from src.processing.newsgroups_data_acquisition import *
from src.preparation.feature_extraction import *
from src.modeling.model_evaluation import *

class DLClassificationTest(unittest.TestCase):
    @unittest.skip("Test ignored (used only experiment)")
    def test_MLP_classifier(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        X_train, X_test = tfidf_transformer(dataset_train, dataset_test)
        # X_train, X_test = count_vectorizer(dataset_train, dataset_test)
        y_train = dataset_train.target
        y_test = dataset_test.target
        mlp_classifier = MLP_Classifier(X_train, y_train, X_test,
                                        y_test)
        model = mlp_classifier.perform_training()
        evaluate_dl_model(model, X_test, y_test)

    @unittest.skip("Test ignored (used only experiment)")
    def test_FC_CNN_classifier(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        X_train = dataset_train.data
        X_test = dataset_test.data
        y_train = dataset_train.target
        y_test = dataset_test.target
        fc_cnn_classifier = FC_CNN_Classifier(X_train, y_train, X_test, y_test)
        fc_cnn_classifier.perform_training()
