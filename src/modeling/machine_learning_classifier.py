from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from src.preparation.feature_extraction import *
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

class MachineLearningClassifier:


    ##TODO: deux contrcteurs  avec des données test + tain.  un autre pour le cross validation
    ##One module can contain several classes ClassName.
    ##TODO : instanciaton doit permettre le choix de l'architecture et le modèle
    # TODO: Cross validation  and data splitting
    def process_training(self, dataset_train, dataset_test):
        """perform training process. To change features  we must update the following lines."""
        # dataset_train, dataset_test = download_newsgroups_dataset()
        #X_train, X_test = tfidf_transformer(dataset_train, dataset_test)
        X_train, X_test = count_vectorizer(dataset_train, dataset_test)
        labels_train = dataset_train.target
        labels_test = dataset_test.target
        lb_encoder = LabelEncoder()
        y = lb_encoder.fit_transform(labels_train)
        y_test = lb_encoder.transform(labels_test)
        y_train = tf.keras.utils.to_categorical(y, num_classes=None, dtype="float32")

        classifier = self.train_ml_model(self, X_train, y_train)
        self.evaluate_model(classifier, lb_encoder, X_test, y_test)

    def train_ml_model(self, X_train, y_train):
        """this method fits and trains the DL model.  In order to select another
        DL architecture, we must change the build_fn parameter"""
        print("train the model ... ")
        classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        classifier.fit(X_train, y_train)
        print(X_train.shape)
        print(y_train.shape)
        classifier.fit(X_train, y_train)
        return classifier

    @staticmethod
    def evaluate_model(classifier,lb_encoder, X_test, y_test):
        y_predict = classifier.predict(X_test)
        print(y_test)
        print(y_predict)
        print(lb_encoder.inverse_transform(y_predict))
        # res = precision_recall_fscore_support(labels_test, y_predict, average='macro')
        # print("[precision, recall, fscore] : " + str(res))
