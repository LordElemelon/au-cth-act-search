import pandas as pd

from backend.services import utils, io, glove, tfidf, model_manager, bert, allen_nlp, sentencebert, infersent
from backend.services.config import Config
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def find_documents_word2vec(query, wv, vec_op=utils.average, tf_idf=False):
    query_tokens = utils.preprocess(query)

    tfidf_scores = None
    if tf_idf:
        tfidf_scores = tfidf.tfidf([path_content[1] for path_content in io.read_documents_for_tfidf()])

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
        if i == Config.sections_to_display:
            break

        with open(df['doc'][i], 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_doc2vec(query, model):
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
        if i == Config.sections_to_display:
            break

        with open(df['doc'][i], 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_tfidf(query):
    documents = io.read_documents_for_tfidf()

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
        if i == Config.sections_to_display:
            break

        with open(df['doc'][i], 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_fasttext(query, wv, vec_op=utils.average):
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
        if i == Config.sections_to_display:
            break

        with open(df['doc'][i], 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


def find_documents_glove(query, vec_op=utils.average):
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
        if i == Config.sections_to_display:
            break

        with open(df['doc'][i], 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections


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

    return {'result': result}


def answer(question, technique):
    result = None

    if technique == 'bert':
        result = bert.bert(question, embd_technique='word2vec')
    elif technique == 'allennlp':
        result = allen_nlp.allennlp(question, embd_technique='word2vec')

    return {'result': result}
