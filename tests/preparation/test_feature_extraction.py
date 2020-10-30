from src.preparation.feature_extraction import *
from src.processing.newsgroups_data_acquisition import *
import unittest


class FeatureExtractionTest(unittest.TestCase):
    @unittest.skip("Test ignored (just for experiment)")
    def test_tfidf_extractor(self):
        train_data_set, test_data_set = download_newsgroups_dataset()
        tfidf_extractor(train_data_set, test_data_set)
