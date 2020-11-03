from src.preparation.feature_extraction import *
from src.processing.newsgroups_data_acquisition import *
import unittest


class FeatureExtractionTest(unittest.TestCase):
    @unittest.skip("Test ignored (used only experiment)")
    def test_tfidf_transformer(self):
        train_data_set, test_data_set = download_newsgroups_dataset()
        tfidf_transformer(train_data_set, test_data_set)

    @unittest.skip("Test ignored (used only experiment)")
    def test_count_vectorizer(self):
        train_data_set, test_data_set = download_newsgroups_dataset()
        count_vectorizer(train_data_set, test_data_set)
