from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def tfidf_extractor(corpus):
    vectorizer = TfidfVectorizer(binary=True)
    tfidf_score = vectorizer.fit_transform(corpus)
    # get tfidf vector for first document
    # first_document_vector = tfidf_score[0]
    # df = pd.DataFrame(first_document_vector.T.todense(), index=vectorizer.get_feature_names(),
    #                   columns=["tfidf"])
    # df.sort_values(by=["tfidf"], ascending=False, inplace=True)
    return tfidf_score
