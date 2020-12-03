import numpy as np
from gensim.models import keyedvectors


def load_glove_indexes(file_name):
    print("loading Glove indexes")
    file = open(file_name, encoding="utf8")
    embeddings_index = dict()
    for line in file:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_index[word] = coefs
    embedding_dim = len(coefs)
    file.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index, embedding_dim


def load_googlenews_indexes(file_name):
    print("loading Google News word2vec")
    embeddings_index = dict()
    model = keyedvectors.load_word2vec_format(fname=file_name, binary=True)
    for word in model.key_to_index.keys():
        embeddings_index[word] = model.get_vector(word).ravel()
    embedding_dim = model.vector_size
    print("embedding_dim :" + str(embedding_dim))
    print("word2vec number of words :" + str(len(embeddings_index)))
    return embeddings_index, embedding_dim
