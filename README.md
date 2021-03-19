# Table of contents
- [Instructions](#instructions)
  - [Before use](#before-use)
    - [Embedding vectors training](#embedding-vectors-training)
  - [Launching the application](#launching-the-application)
- [Team](#team)

# Instructions

## Before use

<ins>Make sure you have all of the following:</ins>

* Python 3.7 with installed requirements
* Trained pipeline (terminal): `python -m spacy download en_core_web_lg`
* Stopwords corpus (terminal): `python -m nltk.downloader stopwords`
* InferSent (terminal on the *app\backend\data\infersent* path): `curl -Lo infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl`
* fastText
  * [vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip) (unzip the embedding vectors into the *app\backend\data\fasttext* folder)
  * [model](https://github.com/facebookresearch/fastText) (clone the model into the *app\backend\data\fasttext* folder [after cloning, there should be a path *app\backend\data\fasttext\fastText*])
* [GloVe](http://nlp.stanford.edu/data/glove.840B.300d.zip) (unzip the embedding vectors into the *app\backend\data\glove* folder)

<ins>Also, make sure you have the following paths:</ins>

* *app\backend\data\corpus\data_by_sect* with preprocessed sections
* *app\backend\data\corpus\data_orig_by_sect* with original sections
* *app\backend\data\corpus\data_orig* with documents (acts)

### Embedding vectors training

To train embedding vectors and calculate document vectors (Word2Vec, Doc2Vec, fastText and GloVe), run the *app\backend\services\train.py* script (all of them will be trained, so comment models you don't want to train).

## Launching the application

To launch the app, in terminal on the *app\backend* path, use one of the following options:
- for development purposes (with auto reload) first run `SET FLASK_ENV=development` and then `flask run`
- in production `flask run`

# Team

* Miloš Radojčin, R2 15/2020 (@milosradojcin)
* Dejan Dokić, R2 17/2020 (@dejandokic)
* Luka Marić, R2 34/2020 (@LordElemelon)