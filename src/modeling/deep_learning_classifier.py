

def create_MLP_model(input):
    train_products = LeclercDataLoader.read_leclerc_dataset(leclerc_dataSet_path)
    print(str(len(train_products)) + " products have been loaded for training")
    train_texts = load_texts(train_products)
    train_categories = load_categories(train_products)
    tokenizer = create_tokenizer(train_texts)

    # Save the tokenizer
    with open('tokenizer_Leclerc.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    x_train = tokenizer.texts_to_matrix(train_texts, mode='tfidf')
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_categories)

    # Save the MultiLabelBinarizer
    with open('MultiLabelBinarizer_Leclerc.pkl', 'wb') as handle:
        pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create the DL model
    model = Sequential()
    model.add(Dense(1000, input_shape=(vocab_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_categories))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=10,
                        verbose=1,
                        validation_split=0.1)
    model.model.save('papud_tfidf_categorizer_Leclerc.h5')
