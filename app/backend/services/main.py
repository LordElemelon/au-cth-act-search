from nltk.corpus import stopwords
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .config import Config

import pandas as pd
from . import allen_nlp, bert, glove, infersent, io_manager, model_manager, sentencebert, tfidf, utils


def find_documents_word2vec(query, wv, vec_op=utils.average, tf_idf=False, basic_search=True):
    query_tokens = utils.preprocess(query)

    tfidf_scores = None
    if tf_idf:
        tfidf_scores = tfidf.tfidf([path_content[1] for path_content in io_manager.read_documents_for_tfidf()])

    query_vec = vec_op([wv[word] * tfidf_scores[word] if tf_idf else wv[word] for word in query_tokens])

    df = pd.DataFrame(columns=['doc', 'similarity'])
    with open('data/word2vec/document_vectors.txt', 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            values = line.split()
            doc_similarity = utils.cosine(query_vec, [float(x) for x in values[1:]])
            df.loc[i] = [values[0], doc_similarity]

    df.sort_values(by=['similarity'], inplace=True, ignore_index=True, ascending=False)

    sections = []
    for i, row in df.iterrows():
        if basic_search:
            if i == Config.sections_to_display:
                break
        elif i == Config.sections_to_display:
            break

        path = df['doc'][i] if basic_search else df['doc'][i].replace('data_orig_by_sect', 'data_by_sect')
        with open(path, 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_doc2vec(query, model, basic_search=True):
    query_tokens = utils.preprocess(query)
    query_vector = model.infer_vector(query_tokens)

    df = pd.DataFrame(columns=['doc', 'similarity'])
    with open('data/doc2vec/document_vectors.txt', 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            values = line.split()
            doc_similarity = utils.cosine(query_vector, [float(x) for x in values[1:]])
            df.loc[i] = [values[0], doc_similarity]

    df.sort_values(by=['similarity'], inplace=True, ignore_index=True, ascending=False)

    sections = []
    for i, row in df.iterrows():
        if basic_search:
            if i == Config.sections_to_display:
                break
        elif i == Config.sections_to_display:
            break

        path = df['doc'][i] if basic_search else df['doc'][i].replace('data_orig_by_sect', 'data_by_sect')
        with open(path, 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_tfidf(query, basic_search=True):
    documents = io_manager.read_documents_for_tfidf()

    tokenizer = tfidf.LemmaTokenizer()
    token_stop = tokenizer(' '.join(stopwords.words('english')))

    # Create TF-IDF model
    vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    doc_vectors = vectorizer.fit_transform([query] + [path_content[1] for path_content in documents])

    # Calculate similarities
    cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[1:]]
    df = pd.DataFrame(columns=['doc', 'similarity'])
    for i, doc in enumerate(documents):
        # print("Doc = ", doc, "; similarity = ", document_scores[i])
        df.loc[i] = [doc[0], document_scores[i]]

    df.sort_values(by=['similarity'], inplace=True, ignore_index=True, ascending=False)

    sections = []
    for i, row in df.iterrows():
        if basic_search:
            if i == Config.sections_to_display:
                break
        elif i == Config.sections_to_display:
            break

        path = df['doc'][i] if basic_search else df['doc'][i].replace('data_orig_by_sect', 'data_by_sect')
        with open(path, 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_fasttext(query, wv, vec_op=utils.average, basic_search=True):
    query_tokens = utils.preprocess(query)
    query_vec = vec_op([wv[word] for word in query_tokens])

    df = pd.DataFrame(columns=['doc', 'similarity'])

    with open('data/fasttext/document_vectors.txt', 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            values = line.split()
            doc_similarity = utils.cosine(query_vec, [float(x) for x in values[1:]])
            df.loc[i] = [values[0], doc_similarity]

    df.sort_values(by=['similarity'], inplace=True, ignore_index=True, ascending=False)

    sections = []
    for i, row in df.iterrows():
        if basic_search:
            if i == Config.sections_to_display:
                break
        elif i == Config.sections_to_display:
            break

        path = df['doc'][i] if basic_search else df['doc'][i].replace('data_orig_by_sect', 'data_by_sect')
        with open(path, 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

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
    df = pd.DataFrame(columns=['doc', 'similarity'])

    with open('data/glove/document_vectors.txt', 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            values = line.split()
            doc_similarity = utils.cosine(query_vec, [float(x) for x in values[1:]])
            df.loc[i] = [values[0], doc_similarity]

    df.sort_values(by=['similarity'], inplace=True, ignore_index=True, ascending=False)

    sections = []
    for i, row in df.iterrows():
        if basic_search:
            if i == Config.sections_to_display:
                break
        elif i == Config.sections_to_display:
            break

        path = df['doc'][i] if basic_search else df['doc'][i].replace('data_orig_by_sect', 'data_by_sect')
        with open(path, 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_lda(query):
    model, topic_repres, belonging = model_manager.load_model(model='lda')
    id2word = corpora.Dictionary.load(Config.lda_path+'/dict.pickle')

    corp = utils.preprocess(query, punct=True, stopwrd=True, lda_clear=True)
    doc_bow = id2word.doc2bow(corp)
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

    df = pd.DataFrame(columns=['doc', 'similarity'])
    with open('data/word2vec/document_vectors.txt', 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            values = line.split()
            if not any((filenm+"\\") in values[0] for filenm in filenames):
                continue
            doc_similarity = utils.cosine(query_vec, [float(x) for x in values[1:]])
            df.loc[i] = [values[0], doc_similarity]

    df.sort_values(by=['similarity'], inplace=True, ignore_index=True, ascending=False)
    print(df)

    sections = []
    for i, row in df.iterrows():
        if i == Config.sections_to_display:
            break

        path = df['doc'][i]
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

if __name__ == '__main__':
    find_documents("illegal gambling", "lda")