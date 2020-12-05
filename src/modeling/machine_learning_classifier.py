from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC

class MachineLearningClassifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_multinomialNB(self):
        estimator = MultinomialNB()
        estimator.fit(self.X_train, self.y_train)
        score = estimator.score(self.X_test, self.y_test)
        print("score multinomialNB：\n", score)

    def train_svm(self):
       estimator = LinearSVC(random_state=0, tol=1e-05)  # Linear Kernel
       estimator.fit(self.X_test, self.y_test)
       score = estimator.score(self.X_test, self.y_test)
       print("svm estimator：\n", score)


