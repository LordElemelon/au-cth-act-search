from gensim.models import KeyedVectors
from gensim.models import Word2Vec, doc2vec as Doc2Vec, FastText
from . import doc2vec, fasttext, utils, word2vec

import gensim.downloader as api
import os


def train(model, transfer_learning=False):
    embd_model = None
    if model == 'word2vec':
        if transfer_learning:
            return word2vec.word2vec_transfer_learning()

        embd_model = word2vec.train_word2vec()
    elif model == 'doc2vec':
        embd_model = doc2vec.train_doc2vec(dm=0, vector_size=50, epochs=50)
    elif model == 'fasttext':
        embd_model = fasttext.train_fasttext()

    save_model(embd_model)

    if model == 'word2vec':
        print('Word2Vec training finished...')
    elif model == 'doc2vec':
        print('Doc2Vec training finished...')
    elif model == 'fasttext':
        print('fastText training finished...')
    return embd_model


def save_model(model):
    if type(model) == Word2Vec:
        path = 'data/word2vec'
        if not os.path.exists(path):
            os.makedirs(path)

        model.wv.save(path + '/gensim-word2vec.wv')
    elif type(model) == Doc2Vec.Doc2Vec:
        path = 'data/doc2vec'
        if not os.path.exists(path):
            os.makedirs(path)

        model.save(path + '/gensim-doc2vec.model')
    elif type(model) == FastText:
        path = 'data/fasttext'
        if not os.path.exists(path):
            os.makedirs(path)

        model.wv.save(path + '/gensim-fasttext.wv')


def load_model(model, pretrained=False):
    if model == 'word2vec':
        if pretrained:
            wv = api.load('word2vec-google-news-300')
            return wv

        path = 'data/word2vec/gensim-word2vec.wv'
        if not os.path.exists(path):
            print(
                f"{utils.Colors.FAIL}You don't have any trained Word2Vec model. Try using the pretrained model (pretrained = True).{utils.Colors.ENDC}")
            return

        return KeyedVectors.load(path, mmap='r')
    elif model == 'doc2vec':
        path = 'data/doc2vec/gensim-doc2vec.model'
        if not os.path.exists(path):
            print(f"{utils.Colors.FAIL}You don't have any trained Doc2Vec model.{utils.Colors.ENDC}")
            return

        return Doc2Vec.Doc2Vec.load(path)
    elif model == 'fasttext':
        if pretrained:
            fin = open('data/fasttext/wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
            n, d = map(int, fin.readline().split())
            data = {}
            for line in fin:
                tokens = line.rstrip().split(' ')
                data[tokens[0]] = map(float, tokens[1:])
            return data

        path = 'data/fasttext/gensim-fasttext.wv'
        if not os.path.exists(path):
            print(
                f"{utils.Colors.FAIL}You don't have any trained fastText model. Try using the pretrained model (pretrained = True).{utils.Colors.ENDC}")
            return

        return KeyedVectors.load(path, mmap='r')
