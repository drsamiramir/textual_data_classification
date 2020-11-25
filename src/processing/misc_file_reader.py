import numpy as np


def load_glove_indexes(file_name):
    file = open(file_name, encoding="utf8")
    embedding_dim = 0
    embeddings_index = dict()
    for line in file:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass

        embeddings_index[word] = coefs
    embedding_dim += len(coefs)
    file.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index, embedding_dim
