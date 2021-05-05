import numpy as np
import pickle
from gensim import corpora
from scipy.sparse import load_npz
from .config import Config
from . import allen_nlp, bert, glove, infersent, io_manager, model_manager, sentencebert, utils


def find_documents_word2vec(query, wv, vec_op=utils.average, basic_search=True):
    query_tokens = utils.preprocess(query)
    query_vec = vec_op([wv[word] for word in query_tokens])

    matrix = np.load('data/word2vec/document_vectors.npy', allow_pickle=True)
    matrix, docs = np.array([pair[1] for pair in matrix]), [pair[0] for pair in matrix]

    results = []
    [results.append((docs[i], utils.cosine(query_vec, matrix[i]))) for i in range(len(docs))]

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:Config.sections_to_display]

    sections = []
    [sections.append(open(row[0] if basic_search else row[0].replace('data_orig_by_sect', 'data_by_sect'), 'r', encoding="utf8").read()) for row in results]

    return sections


def find_documents_doc2vec(query, model, basic_search=True):
    query_tokens = utils.preprocess(query)
    query_vec = model.infer_vector(query_tokens)

    matrix = np.load('data/doc2vec/document_vectors.npy', allow_pickle=True)
    matrix, docs = np.array([pair[1] for pair in matrix]), [pair[0] for pair in matrix]

    results = []
    [results.append((docs[i], utils.cosine(query_vec, matrix[i]))) for i in range(len(docs))]

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:Config.sections_to_display]

    sections = []
    [sections.append(open(row[0] if basic_search else row[0].replace('data_orig_by_sect', 'data_by_sect'), 'r', encoding="utf8").read()) for row in results]

    return sections


def find_documents_tfidf(query, basic_search=True):
    tokens = ' '.join(utils.preprocess(query))
    paths = io_manager.get_paths()

    vectorizer = pickle.load(open('data/tfidf/vectorizer.pk', 'rb'))
    document_vectors = load_npz('data/tfidf/document_vectors.npz')

    vec = vectorizer.transform([tokens])

    results = []
    [results.append((paths[i], utils.cosine(document_vectors[i].toarray(), vec.toarray().transpose()))) for i in range(len(paths))]

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:Config.sections_to_display]

    sections = []
    [sections.append(open(row[0] if basic_search else row[0].replace('data_orig_by_sect', 'data_by_sect'), 'r', encoding="utf8").read()) for row in results]

    return sections


def find_documents_fasttext(query, wv, vec_op=utils.average, basic_search=True):
    query_tokens = utils.preprocess(query)
    query_vec = vec_op([wv[word] for word in query_tokens])

    matrix = np.load('data/fasttext/document_vectors.npy', allow_pickle=True)
    matrix, docs = np.array([pair[1] for pair in matrix]), [pair[0] for pair in matrix]

    results = []
    [results.append((docs[i], utils.cosine(query_vec, matrix[i]))) for i in range(len(docs))]

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:Config.sections_to_display]

    sections = []
    [sections.append(open(row[0] if basic_search else row[0].replace('data_orig_by_sect', 'data_by_sect'), 'r', encoding="utf8").read()) for row in results]

    return sections


def find_documents_glove(query, vec_op=utils.average, basic_search=True):
    embeddings_dict = glove.load_glove()
    query_tokens = utils.preprocess(query)

    query_vec = []
    for word in query_tokens:
        try:
            query_vec.append(embeddings_dict[word])
        except:
            pass

    query_vec = vec_op(query_vec)

    matrix = np.load('data/glove/document_vectors.npy', allow_pickle=True)
    matrix, docs = np.array([pair[1] for pair in matrix]), [pair[0] for pair in matrix]

    results = []
    [results.append((docs[i], utils.cosine(query_vec, matrix[i]))) for i in range(len(docs))]

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:Config.sections_to_display]

    sections = []
    [sections.append(open(row[0] if basic_search else row[0].replace('data_orig_by_sect', 'data_by_sect'), 'r', encoding="utf8").read()) for row in results]

    return sections


def find_documents_lda(query):
    model, topic_repres, belonging = model_manager.load_model(model='lda')
    id2word = corpora.Dictionary.load(Config.lda_path+'/dict.pickle')

    corp = utils.preprocess(query, lda_clear=True)
    doc_bow = id2word.doc2bow(corp, return_missing=True)[0]
    all_belong_values = model[doc_bow]
    max_belong_value = max([y for x, y in all_belong_values])
    filtered_belong_values = [x for x, y in all_belong_values if y > 0.5*max_belong_value]

    print(all_belong_values)
    print(filtered_belong_values)

    filenames = list(set().union(*[belonging[x] for x in filtered_belong_values]))
    print(len(filenames))
    print(filenames)

    word2vec_wv = model_manager.load_model(model='word2vec')

    query_vec = utils.average([word2vec_wv[word] for word in corp])

    results = []
    with open('data/word2vec/document_vectors.txt', 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            values = line.split()
            if not any((filenm+"\\") in values[0] for filenm in filenames):
                continue
            doc_similarity = utils.cosine(query_vec, [float(x) for x in values[1:]])
            results.append((values[0], doc_similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:Config.sections_to_display]

    sections = []
    for row in results:
        path = row[0]
        with open(path, 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def read_sections(names):
    return {'result': io_manager.read_sections(names)}


def find_documents(query, technique):
    result = None

    if technique == 'word2vec':
        word2vec_wv = model_manager.load_model(model='word2vec')

        result = find_documents_word2vec(query, word2vec_wv)
    elif technique == 'doc2vec':
        doc2vec_model = model_manager.load_model(model='doc2vec')

        result = find_documents_doc2vec(query, doc2vec_model)
    elif technique == 'fasttext':
        fasttext_wv = model_manager.load_model(model='fasttext')

        result = find_documents_fasttext(query, fasttext_wv)
    elif technique == 'tfidf':
        result = find_documents_tfidf(query)
    elif technique == 'glove':
        result = find_documents_glove(query)
    elif technique == 'sentencebert':
        result = sentencebert.sentencebert(query)
    elif technique == 'infersent':
        result = infersent.infersent(query)
    elif technique == 'lda':
        result = find_documents_lda(query)

    return {'result': result}


def answer(question, technique):
    result = None

    if technique == 'bert':
        result = bert.bert(question, embd_technique='glove')
    elif technique == 'allennlp':
        result = allen_nlp.allennlp(question, embd_technique='word2vec')

    return {'result': result}
