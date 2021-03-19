from gensim.models import doc2vec
from backend.services import io


# https://radimrehurek.com/gensim_3.8.3/models/doc2vec.html
def train_doc2vec(dm=1, vector_size=100, window=5, min_count=2, workers=8, epochs=10):
    model = doc2vec.Doc2Vec(dm=dm, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                            epochs=epochs)

    train_corpus = list(io.read_documents_for_doc2vec())

    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    documents_tokens = io.read_documents_for_doc2vec(tokens_only=True)

    with open('../data/doc2vec/document_vectors.txt', 'w', encoding="utf8") as f:
        for doc in documents_tokens:
            doc_vec = model.infer_vector(doc[1])
            f.write(doc[0].strip().replace('data_by_sect', 'data_orig_by_sect')[3:] + ' ' + ' '.join(map(str, doc_vec)) + '\n')

    return model
