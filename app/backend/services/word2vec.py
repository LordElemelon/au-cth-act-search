from gensim.models import Word2Vec
from . import io_manager, tfidf, utils
from pathlib import Path


# https://radimrehurek.com/gensim_3.8.3/models/word2vec.html


def train_word2vec(vec_op=utils.average, tf_idf=False, size=100, window=5, min_count=1, workers=4, sg=0, epochs=5):
    documents_tokens = io_manager.read_documents_for_word2vec()

    sentences = [t[1] for t in documents_tokens]
    model = Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=workers, sg=sg)

    tfidf_scores = None
    if tf_idf:
        tfidf_scores = tfidf.tfidf([path_content[1] for path_content in io_manager.read_documents_for_tfidf()])

    Path("data/word2vec").mkdir(parents=True, exist_ok=True)
    with open('data/word2vec/document_vectors.txt', 'w', encoding="utf8") as f:
        for doc in documents_tokens:
            doc_vec = vec_op([model.wv[word] * tfidf_scores[word] if tf_idf else model.wv[word] for word in doc[1]])
            f.write(doc[0].strip().replace('data_by_sect', 'data_orig_by_sect') + ' ' + ' '.join(map(str, doc_vec)) + '\n')

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
