from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tfidf_transformer(dataset_train, dataset_test):
    tfidf_vectorizer = TfidfVectorizer(binary=True, max_features=100000, stop_words='english')
    X_train = tfidf_vectorizer.fit_transform(dataset_train.data).astype('float32').toarray()
    X_test = tfidf_vectorizer.transform(dataset_test.data).astype('float32').toarray()
    return X_train, X_test


def count_vectorizer(dataset_train, dataset_test):
    count_vectorizer = CountVectorizer(max_features=100000, stop_words='english')
    X_train = count_vectorizer.fit_transform(dataset_train.data).astype('float32').toarray()
    X_test = count_vectorizer.transform(dataset_test.data).astype('float32').toarray()
    return X_train, X_test
