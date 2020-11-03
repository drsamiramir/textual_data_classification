from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


## TODO : instanciaton doit permettre le choix de l'architecture et le modèle
# TODO : Cross validation  and data splitting

class DeepLearningClassifier:
    def __init__(self, input_dim, output_dim, X_train, y_train):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X_train = X_train
        self.y_train = y_train

    def process_training(self):
        label_binarizer = LabelBinarizer()
        y_train_binary = label_binarizer.fit_transform(self.y_train)
        classifier = self.train_dl_model(y_train_binary)
        return classifier

    def train_dl_model(self, y_train_binary):
        """this method fits and trains the DL model.  In order to select another
        DL architecture, we must change the build_fn parameter"""
        print("train the model ... ")
        classifier = KerasClassifier(build_fn=self.build_mlp_architecture, epochs=15,
                                     batch_size=200)
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
