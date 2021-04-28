from gensim.models import Word2Vec
from . import io_manager, tfidf, utils
from pathlib import Path
import numpy as np


# https://radimrehurek.com/gensim_3.8.3/models/word2vec.html
def train_word2vec(vec_op=utils.average, size=100, window=5, min_count=1, workers=4, sg=0, epochs=5):
    documents_tokens = io_manager.read_documents_for_word2vec()

    sentences = [t[1] for t in documents_tokens]
    model = Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=workers, sg=sg)

    Path("data/word2vec").mkdir(parents=True, exist_ok=True)

    matrix = []
    [matrix.append((doc[0].strip().replace('data_by_sect', 'data_orig_by_sect'), vec_op([model.wv[word] for word in doc[1]]))) for doc in documents_tokens]

    matrix = np.asarray(matrix, dtype=object)
    np.save('data/word2vec/document_vectors.npy', matrix)

    return model


def word2vec_transfer_learning():
    sentences = [t[1] for t in io_manager.read_documents_for_word2vec()]

    # size option needs to be set to 300 to be the same as Google's pre-trained model
    model = Word2Vec(size=300, window=5, min_count=1, workers=4, sg=0)
    model.build_vocab(sentences)

    # assign the vectors to the vocabs that are in Google's pre-trained model and your sentences defined above.
    # lockf needs to be set to 1.0 to allow continued training.
    model.intersect_word2vec_format('data/word2vec/GoogleNews-vectors-negative300.bin', lockf=1.0, binary=True)

    # continue training with you own data
    model.train(sentences, total_examples=len(sentences), epochs=5)

    return model
