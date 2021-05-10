import spacy
import numpy as np
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 35000000


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def average(vectors):
    return sum(vectors) / len(vectors)


def punctuation():
    return [',', '.', '\'', '\"', '?', '!', '_', '-', ':', ';', '(', ')', '[', ']', '{', '}',
            '*', '+', '\\', '&', '^', '%', '#', '$', '@', '<', '>', '‘', '`', '~', 'Ă']


def lda_clear_words():
    return ['section', 'act', 'subsection', 'purpose', 'person', 'relation', 'apply', 'provision', '--',
            'paragraph', 'period', 'day', 'give', 'include', 'notice', 'time', 'mean', 'note', 'specify',
            'commonwealth', 'application', 'information', 'relate', 'take', 'require', 'matter', 'respect',
            'provide', 'effect', 'year', 'minister', 'state', 'decision', 'meaning']
            # 'subject', 'division', 'relevant', 'particular', 'refer', 'determine', 'accordance', 'write', 'follow', 'reference', 'satisfied', 'make']
            # 'subject', 'division', 'relevant', 'particular', 'refer']


def preprocess(s, op=nlp, lda_clear=False):
    doc = op(s, disable=['ner', 'parser'])
    stop_words = stopwords.words('english')
    drop_words = lda_clear_words() if lda_clear else []

    tokens = []
    for token in doc:
        if token.lemma_.lower() in drop_words:
            continue
        if lda_clear and (token.lemma_[0].isnumeric() or len(token.lemma_) < 2):
            continue
        if (not token.is_stop and token.text.lower() not in stop_words) and token.text not in punctuation():
            tokens.append(token.lemma_.lower() if token.lemma_ != '-PRON-' else token.text.lower())

    return tokens
