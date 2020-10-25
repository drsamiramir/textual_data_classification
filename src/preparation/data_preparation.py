import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import re


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens


def is_stopword(word):
    """check whether the word is a stopword or not. Basically,
    this method uses the NLTK stopwords and the list of stopwords
    provided by the user"""
    raw_stopword_list = stopwords.words('english')
    lines = [line.rstrip('\n').strip() for line in open('./data/stopwords.txt', encoding='utf8')]
    raw_stopword_list = lines + raw_stopword_list
    if word.lower() in raw_stopword_list:
        return True

    return False


def stem_words(words):
    """stems the word list using English Stemmer"""
    stemmer = EnglishStemmer()
    stemmed_words = list()
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def remove_spacial_characters(text):
    """remove special chacaters from the text"""
    pattern = re.compile('[\W_]+')
    cleaned_text = pattern.sub(' ', text)
    return cleaned_text
