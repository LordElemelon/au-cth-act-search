from gensim.models import doc2vec
from .config import Config

from . import utils
import os


def read_files_walk():
    #print("\n\nHELLO\n\n"+os.path.abspath(Config.corpus_path)+"\n\n")
    
    for root, dirs, files in os.walk(Config.corpus_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), encoding="utf8") as f:
                    yield os.path.join(root, file), f.read()


def read_documents_for_word2vec():
    examples = []
    for path, content in read_files_walk():
        tokens = utils.preprocess(content)

        examples.append((path, tokens))

    return examples


def read_documents_for_doc2vec(tokens_only=False):
    for i, path_content in enumerate(read_files_walk()):
        tokens = utils.preprocess(path_content[1])

        if tokens_only:
            yield path_content[0], tokens
        else:
            yield doc2vec.TaggedDocument(tokens, [i])


def read_documents_for_tfidf():
    corpus = []

    for path, content in read_files_walk():
        corpus.append((path, content))

    return corpus


def read_sections(names):
    sections = []
    for name in names:
        act_sect = name.split('=')
        with open('data/corpus/data_orig_by_sect/' + act_sect[0] + '/' + act_sect[1] + '.txt', 'r', encoding='utf8') as f:
            sections.append(f.read())

    return sections


def merge_all_sections():
    with open('data/corpus/sec_corpus.txt', 'w', encoding="utf8") as f:
        for path, content in read_files_walk():
            f.write(content)
