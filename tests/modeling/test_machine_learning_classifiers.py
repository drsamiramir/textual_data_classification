import unittest
from src.modeling.machine_learning_classifier import *
from src.processing.newsgroups_data_acquisition import *
from sklearn.model_selection import train_test_split


class MLClassificationTest(unittest.TestCase):
    def test_ml_classifiers(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        X_train, X_test = tfidf_transformer(dataset_train, dataset_test)
        #X_train, X_test = count_vectorizer(dataset_train, dataset_test)
        y_train = dataset_train.target
        y_test = dataset_test.target
        ml_classifier = MachineLearningClassifier(X_train, y_train, X_test, y_test)
        ml_classifier.train_multinomialNB()
        ml_classifier.train_linearSVC()


