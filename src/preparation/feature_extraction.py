from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_transformer(dataset_train, dataset_test):
    tfidf_vectorizer = TfidfVectorizer(binary=True, max_features=100000, stop_words='english')
    X_train = tfidf_vectorizer.fit_transform(dataset_train.data).astype('float32')
    X_test = tfidf_vectorizer.transform(dataset_test.data).astype('float32')
    return X_train, X_test
