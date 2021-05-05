import numpy as np
import itertools as it
import pickle
from . import io_manager
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


def train_tfidf():
    # vocabulary = set()
    # documents_tokens = io_manager.read_documents_for_word2vec()
    #
    # [vocabulary.update(set(tokens)) for path, tokens in documents_tokens]
    # vocabulary = sorted(vocabulary)
    #
    # c = Counter(it.chain(*map(set, [tokens for path, tokens in documents_tokens])))
    # idfs_ = np.array([1 + np.log(len(documents_tokens) / c[word]) for word in vocabulary], dtype=np.float16)
    # idfs = [(vocabulary[i], idfs_[i]) for i in range(len(vocabulary))]
    #
    # tfidfs = [(path.strip().replace('data_by_sect', 'data_orig_by_sect'), np.array(np.multiply([tokens.count(word) / len(tokens) for word in vocabulary], idfs_), dtype=np.float16)) for path, tokens in documents_tokens]
    # for path, tokens in documents_tokens:
    #     tfidf_vec = [tokens.count(word) / len(tokens) for word in vocabulary]
    #     tfidf_vec = np.multiply(tfidf_vec, idfs)
    #
    #     tfidfs.append((path.strip().replace('data_by_sect', 'data_orig_by_sect'), np.array(tfidf_vec, dtype=np.float16)))
    #
    # Path('data/tfidf').mkdir(parents=True, exist_ok=True)
    #
    # idfs = np.asarray(idfs, dtype=object)
    # np.save('data/tfidf/idf_vectors.npy', idfs)
    #
    # tfidfs = np.asarray(tfidfs, dtype=object)
    # np.save('data/tfidf/document_vectors.npy', tfidfs)

    documents_tokens = io_manager.read_documents_for_word2vec()
    tokens = [' '.join(t) for path, t in documents_tokens]

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(tokens)

    save_npz('data/tfidf/document_vectors.npz', doc_vectors)

    return vectorizer
