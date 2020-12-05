from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preparation.feature_extraction import *


class MachineLearningClassifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_multinomialNB(self):
        estimator = MultinomialNB()
        estimator.fit(self.X_train, self.y_train)
        y_predict = estimator.predict(self.X_test)
        print("y_predict:\n", y_predict)
        score = estimator.score(self.X_test, self.y_test)
        print("score multinomialNB：\n", score)


    def train_linearSVC(self):
        estimator = LinearSVC()
        estimator.fit(self.X_train, self.y_train)
        y_predict = estimator.predict(self.X_test)
        print("y_predict:\n", y_predict)
        score = estimator.score(self.X_test, self.y_test)
        print("score linearSVC：\n", score)
