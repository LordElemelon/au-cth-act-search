import numpy as np

from . import utils, io_manager


def load_glove(dim='50'):
    embeddings_dict = {}
    with open(f"../data/glove/glove.840B.300d.txt", 'r', encoding="utf8") as f:  # glove.6B.{dim}d
        for line in f:
            values = line.split()
            try:
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            except:
                pass

    return embeddings_dict


def calculate_documents_glove(vec_op=utils.average):
    embeddings_dict = load_glove()
    documents_tokens = io_manager.read_documents_for_word2vec()

    with open('../data/glove/document_vectors.txt', 'w', encoding="utf8") as f:
        for doc in documents_tokens:
            doc_vec = []
            for word in doc[1]:
                try:
                    doc_vec.append(embeddings_dict[word])
                except:
                    pass

            doc_vec = vec_op(doc_vec)
            f.write(doc[0].strip().replace('data_by_sect', 'data_orig_by_sect')[3:] + ' ' + ' '.join(map(str, doc_vec)) + '\n')
