from sklearn.feature_extraction.text import TfidfVectorizer
from . import utils


class LemmaTokenizer:
    def __call__(self, doc):
        return utils.preprocess(doc)


def tfidf(corpus):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    matrix = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names()
    feature_index = matrix[0, :].nonzero()[1]
    tfidf_scores = dict(zip([feature_names[i] for i in feature_index], [matrix[0, x] for x in feature_index]))

    for name in feature_names:
        if name not in tfidf_scores:
            tfidf_scores[name] = 0.0

    return tfidf_scores


def tf(document):
    tokens = utils.preprocess(document)

    tf_dict = {}
    for word in tokens:
        tf_dict[word] = tokens.count(word) / len(tokens)

    return tf_dict


def corr_matrix():
    """
    Just an example...
    https://ted-mei.medium.com/demystify-tf-idf-in-indexing-and-ranking-5c3ae88c3fa0
    """
    inception = "A young man, exhausted and delirious, washes up on a beach, looking up momentarily to see two young children..."
    shutter_island = "U.S. Marshals Teddy Daniels (Leonardo DiCaprio) and Chuck Aule (Mark Ruffalo) are on a ferryboat..."

    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform([inception, shutter_island])

    # correlation matrix between the documents
    matrix = (vecs * vecs.T).A
