from src.processing.newsgroups_data_acquisition import *
import unittest


class NewsgroupAcquisitionTest(unittest.TestCase):

    @unittest.skip("Test ignored (used only for experiment)")
    def test_download_newsgroups_dataset(self):
        dataset_train, dataset_test = download_newsgroups_dataset()
        self.assertTrue(len(dataset_train.data) > 100 and len(dataset_test.data) > 100)
