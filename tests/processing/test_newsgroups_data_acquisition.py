from src.processing.newsgroups_data_acquisition import *
import unittest


class NewsgroupAcquistitionTest(unittest.TestCase):

    def test_download_dataset(self):
        download_dataset()
