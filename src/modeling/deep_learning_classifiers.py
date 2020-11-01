import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from src.preparation.feature_extraction import *


class DeepLearningClassifier:
    def __init__(self, architecture):
        self.architecture = architecture

    ##One module can contain several classes ClassName.
    ##TODO: formuler des classes
    ##TODO : instanciaton doit permettre le choix de l'architecture et le modèle
    # TODO: Cross validation  and data splitting
    def process_training(self, dataset_train, dataset_test):
        """perform training process. To change features  we must update the following lines."""
        #dataset_train, dataset_test = download_newsgroups_dataset()
        X_train, X_test = tfidf_transformer(dataset_train, dataset_test)
        labels_train = dataset_train.target
        labels_test = dataset_test.target
        lb_encoder = LabelEncoder()
        y = lb_encoder.fit_transform(labels_train)
        y_train = tf.keras.utils.to_categorical(y, num_classes=None, dtype="float32")
        classifier = self.train_dl_model(self, X_train, y_train)
        print("model evaluation... ")


    def train_dl_model(self, X_train, y_train):
        """this method fits and trains the DL model.  In order to select another
        DL architecture, we must change the build_fn parameter"""
        print("train the model ... ")
        classifier = KerasClassifier(build_fn=self.build_mlp_architecture, epochs=10, batch_size=140)
        print(X_train.shape)
        print(y_train.shape)
        classifier.fit(X_train, y_train)
        return classifier

    @staticmethod
    def build_mlp_architecture():
        model = Sequential()
        model.add(Dense(256, input_dim=100000, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(20, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
