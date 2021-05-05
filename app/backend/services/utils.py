import spacy
import numpy as np
import nltk
from nltk.corpus import stopwords

# Uncomment the next line just for the first run, then comment it
# nltk.download('punkt')
nlp = spacy.load('en_core_web_lg')
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


def add(vectors):
    return sum(vectors)


def average(vectors):
    return add(vectors) / len(vectors)


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


# If not lemmatization, then tokenize the string, remove the punctuation, or just simply call utils.simple_preprocess(line)
def preprocess(s, op=nlp, lowercase=True, punct=False, stopwrd=False, lda_clear=False):
    """
    lowercase -- convert words to lowercase, default = True
    punct -- get rid of punctuation, default = False
    """
    doc = op(s, disable=['ner', 'parser'])
    stop_words = stopwords.words('english') if stopwrd else []
    drop_words = lda_clear_words() if lda_clear else []

    tokens = []
    for token in doc:
        if token.text.lower() in stop_words:
            continue
        if token.lemma_.lower() in drop_words:
            continue
        if lda_clear and (token.lemma_[0].isnumeric() or len(token.lemma_) < 2):
            continue
        if punct:
            if not token.is_stop and token.text not in punctuation():
                tokens.append(
                    (token.lemma_.lower() if lowercase else token.lemma_) if token.lemma_ != '-PRON-' else (
                        token.text.lower() if lowercase else token.text))
        else:
            tokens.append((token.lemma_.lower() if lowercase else token.lemma_) if token.lemma_ != '-PRON-' else (
                token.text.lower() if lowercase else token.text))

    return tokens
