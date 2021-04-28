from gensim.models import doc2vec
from . import io_manager
from pathlib import Path
import numpy as np


# https://radimrehurek.com/gensim_3.8.3/models/doc2vec.html
def train_doc2vec(dm=1, vector_size=100, window=5, min_count=2, workers=8, epochs=10):
    model = doc2vec.Doc2Vec(dm=dm, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                            epochs=epochs)

    train_corpus = list(io_manager.read_documents_for_doc2vec())

    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    documents_tokens = io_manager.read_documents_for_doc2vec(tokens_only=True)

    Path("data/doc2vec").mkdir(parents=True, exist_ok=True)

    matrix = []
    [matrix.append((doc[0].strip().replace('data_by_sect', 'data_orig_by_sect'), model.infer_vector(doc[1]))) for doc in documents_tokens]

    matrix = np.asarray(matrix, dtype=object)
    np.save('data/doc2vec/document_vectors.npy', matrix)

    return model
