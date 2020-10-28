from src.preparation.feature_extraction import *
import unittest


class FeatureExtractionTest(unittest.TestCase):
    #@unittest.skip("Test ignored (just for experiment)")
    def test_tfidf_extractor(self):
        corpus = ["the house had a tiny little mouse",
                  "the cat saw the mouse",
                  "the mouse ran away from the house",
                  "the cat finally ate the mouse",
                  "the end of the mouse story"
                  ]
        print(tfidf_extractor(corpus)[0])
