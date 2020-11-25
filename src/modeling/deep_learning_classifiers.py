from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, MaxPooling1D, Conv1D, Dropout
from src.processing.misc_file_reader import *
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer


## TODO : instanciaton doit permettre le choix de l'architecture et le modèle
# TODO : Cross validation  and data splitting

class MLP_Classifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.input_dim = X_train.shape[1]
        self.output_dim = len(np.unique(y_train))
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
        """this method fits and trains the DL model.  In order to select another
        DL architecture, we must change the build_fn parameter"""
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
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model


class FC_CNN_Classifier:
    def __init__(self, X_train, y_train, X_test, y_test, MAX_NB_WORDS=20000, MAX_SEQUENCE_LENGTH=1000):
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.embeddings_index, self.embedding_dim = load_glove_indexes("./data/glove.6B.100d.txt")
        self.word_index = None

    def perform_training(self):
        label_binarizer = LabelBinarizer()
        y_train_binary = label_binarizer.fit_transform(self.y_train)
        y_test_binary = label_binarizer.transform(self.y_test)
        classifier = self.train_dl_model(y_train_binary, y_test_binary)
        return classifier

    def process_data(self):
        text = np.concatenate((self.X_train, self.X_test), axis=0)
        text = np.array(text)
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        self.word_index = tokenizer.word_index
        text = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        print('Found %s unique tokens.' % len(self.word_index))
        X_train_num = text[0:len(self.X_train), ]
        X_test_num = text[len(self.X_train):, ]
        return X_train_num, X_test_num

    def train_dl_model(self, y_train_binary, y_test_binary):
        """this method fits and trains the DL model.  In order to select another"""
        X_train_num, X_test_num = self.process_data()
        classifier = KerasClassifier(build_fn=self.build_fc_cnn_architecture, epochs=15,
                                     batch_size=100, validation_data=(X_test_num, y_test_binary))
        classifier.fit(X_train_num, y_train_binary)
        return classifier

    def build_fc_cnn_architecture(self):
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                if len(embedding_matrix[i]) != len(embedding_vector):
                    print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                          "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                    " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                    exit(1)
                embedding_matrix[i] = embedding_vector
        embedding_layer = Embedding(len(self.word_index) + 1, self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        sequence_input = Input(shape=(1000,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(20, activation='softmax')(x)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
