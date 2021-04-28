from . import utils, io_manager
import numpy as np


def load_glove():
    embeddings_dict = {}
    with open(f"data/glove/glove.6B.100d.txt", 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            try:
                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float16)
                embeddings_dict[word] = vector
            except:
                pass

    return embeddings_dict


def calculate_documents_glove(vec_op=utils.average):
    embeddings_dict = load_glove()
    documents_tokens = io_manager.read_documents_for_word2vec()

    matrix = []
    for doc in documents_tokens:
        doc_vec = []
        for word in doc[1]:
            try:
                doc_vec.append(embeddings_dict[word])
            except:
                pass

        matrix.append((doc[0].strip().replace('data_by_sect', 'data_orig_by_sect'), vec_op(doc_vec)))

    matrix = np.asarray(matrix, dtype=object)
    np.save('data/glove/document_vectors.npy', matrix)

    print('GloVe calculation finished...\n')
