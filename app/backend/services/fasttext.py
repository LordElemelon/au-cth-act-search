from gensim.models import FastText
from . import utils, io_manager
from pathlib import Path
import numpy as np


def train_fasttext(vec_op=utils.average):
    model = FastText()

    # build the vocabulary
    model.build_vocab(corpus_file='data/corpus/sec_corpus.txt')

    # train the model
    model.train(corpus_file='data/corpus/sec_corpus.txt', epochs=model.epochs,
                total_examples=model.corpus_count, total_words=model.corpus_total_words)

    documents_tokens = io_manager.read_documents_for_word2vec()

    Path("data/fasttext").mkdir(parents=True, exist_ok=True)

    matrix = []
    [matrix.append((doc[0].strip().replace('data_by_sect', 'data_orig_by_sect'), vec_op([model.wv[word] for word in doc[1]]))) for doc in documents_tokens]

    matrix = np.asarray(matrix, dtype=object)
    np.save('data/fasttext/document_vectors.npy', matrix)

    return model
