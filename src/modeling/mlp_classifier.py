from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class MLP_Classifier:
    """Multilayer perceptron based classification.  This class should take numerical values as input
    (X_train, X_test).  A feature engineering step must be performed before (e.g. tfidf, count, ..)"""

    def __init__(self, X_train, y_train, X_test, y_test):
        self.input_dim = X_train.shape[1]
        self.nb_classes = len(np.unique(y_train))
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def perform_training(self):
        label_binarizer = LabelBinarizer()
        y_train_binary = label_binarizer.fit_transform(self.y_train)
        y_test_binary = label_binarizer.transform(self.y_test)
        classifier = self.train_dl_model(y_train_binary, y_test_binary)
        return classifier

    def train_dl_model(self, y_train_binary, y_test_binary):
        print("train the model ... ")
        classifier = KerasClassifier(build_fn=self.build_mlp_architecture, epochs=15,
                                     batch_size=200, validation_data=(self.X_test, y_test_binary))
        print(type(self.X_train))
        classifier.fit(self.X_train, y_train_binary)
        return classifier

    def build_mlp_architecture(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
