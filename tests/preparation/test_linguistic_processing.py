from src.preparation.linguistic_processing import *
import unittest


class LinguisticProcessingTest(unittest.TestCase):

    @unittest.skip("Test ignored (used only for experiment)")
    def test_tokenize(self):
        sentence = "The official home of the Python Programming Language."
        tokens = tokenize(sentence)
        self.assertEqual(tokens,
                         ['The', 'official', 'home', 'of', 'the', 'Python', 'Programming', 'Language', '.'])

    @unittest.skip("Test ignored (used only for experiment)")
    def test_is_stop_word(self):
        self.assertTrue(is_stopword("haha"))
        self.assertTrue(is_stopword("are"))
        self.assertFalse(is_stopword("love"))

    @unittest.skip("Test ignored (used only experiment)")
    def test_stem_words(self):
        text = "The official home of the Python Programming Language"
        self.assertEqual(stem_words(tokenize(text)),
                         ['the', 'offici', 'home', 'of', 'the', 'python', 'program', 'languag'])

    @unittest.skip("Test ignored (used only experiment)")
    def test_remove_special_character(self):
        text = "Hello 2  @  dd  === !! 46 mm/s"
        self.assertEqual(remove_spacial_characters(text), "Hello 2 dd 46 mm s")
