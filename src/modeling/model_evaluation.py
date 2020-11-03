from sklearn.metrics import precision_recall_fscore_support


def evaluate_dl_model(classifier, X_test, y_test):
    y_predict = classifier.predict(X_test)
    eval_score = precision_recall_fscore_support(y_test, y_predict, average='macro')
    print("[precision, recall, fscore] : " + str(eval_score))
