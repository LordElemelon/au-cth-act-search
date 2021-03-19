import spacy
import numpy as np

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


# If not lemmatization, then tokenize the string, remove the punctuation, or just simply call utils.simple_preprocess(line)
def preprocess(s, op=nlp, lowercase=True, punct=False):
    """
    lowercase -- convert words to lowercase, default = True
    punct -- get rid of punctuation, default = False
    """
    doc = op(s, disable=['ner', 'parser'])

    tokens = []
    for token in doc:
        if punct:
            if not token.is_stop and token.text not in punctuation():
                tokens.append(
                    (token.lemma_.lower() if lowercase else token.lemma_) if token.lemma_ != '-PRON-' else (
                        token.text.lower() if lowercase else token.text))
        else:
            tokens.append((token.lemma_.lower() if lowercase else token.lemma_) if token.lemma_ != '-PRON-' else (
                token.text.lower() if lowercase else token.text))

    return tokens
