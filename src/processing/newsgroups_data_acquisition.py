from sklearn.datasets import fetch_20newsgroups
from pprint import pprint


def download_newsgroups_dataset():
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    return newsgroups_train,newsgroups_test





