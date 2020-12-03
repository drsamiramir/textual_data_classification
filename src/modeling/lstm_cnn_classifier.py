import numpy
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
import numpy as np
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


class LSTM_CNN_Classifier:
    def __init__(self, X_train, y_train, X_test, y_test, MAX_NB_WORDS=5000, MAX_SEQUENCE_LENGTH=500,
                 EMBEDDING_VECTOR_LENGTH=32):
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_VECTOR_LENGTH = EMBEDDING_VECTOR_LENGTH
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.nb_classes = len(np.unique(y_train))

    def process_data(self):
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=self.MAX_SEQUENCE_LENGTH)
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=self.MAX_SEQUENCE_LENGTH)

    def train_dl_model(self):
        self.process_data()
        classifier = KerasClassifier(build_fn=self.build_lstm_cnn_architecture, epochs=5,
                                     batch_size=128, validation_data=(self.X_test, self.y_test))
        classifier.fit(self.X_train, self.y_train)

    def build_lstm_cnn_architecture(self):
        model = Sequential()
        model.add(Embedding(self.MAX_NB_WORDS, self.EMBEDDING_VECTOR_LENGTH, input_length=self.MAX_SEQUENCE_LENGTH))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(self.X_train, self.y_train, epochs=3, batch_size=64, validation_data=(self.X_test, self.y_test))
