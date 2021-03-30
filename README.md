# Table of contents
- [Instructions](#instructions)
  - [Before use](#before-use)
    - [Embedding vectors training](#embedding-vectors-training)
    - [LDA topic model training](#lda-topic-model-training)
    - [Elasticsearch](#elasticsearch)
    - [Frontend application](#frontend-application)
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

To train word embedding vectors and calculate document vectors (Word2Vec, Doc2Vec, fastText and GloVe), on the *app\backend* path run the command `python -m services.train`.

### LDA topic model training

To train the LDA model and the corresponding objects used in topic modelling, on the *app\backend* path run the command `python -m services.train_lda`.

### Elasticsearch

<ins>Installation process:</ins>
* Download Elasticsearch from https://www.elastic.co/downloads/elasticsearch
* Download Kibana tool from https://www.elastic.co/downloads/kibana
* Navigate to downloaded elasticsearch folder and inside config/elasticsearch.yml (under "Network" section) add the following lines: (`http.cors.enabled: true` and  `http.cors.allow-origin: "*"`)
* Start elasticsearch by navigating into downloaded elasticsearch directory and run bin\elasticsearch.bat (default port is `9200`)
* Start kibana by navigating into downloaded kibana directory and run bin\kibana.bat (default port is `5601`)
* Open http://localhost:5601 and under `Dev Tools` insert first query from  app/elasticsearch/commands.txt. Inserted query is responsible for defining `law` index.
* Run python script located at app/elasticsearch/json_gen.py in order to generate 2 json files which holds data
* Insert those data into a previously created index by running 2 post requests `curl -H "Content-Type: application/x-ndjson" -XPOST http://localhost:9200/law/_bulk --data-binary "@data.json"`, one for each json file
* Congrats, elasticsearch is now ready for use

### Frontend application
* Navigate into app/frontend directory and run `npm install` and `npm start`

## Launching the application

To launch the app, in terminal on the *app\backend* path, use one of the following options:
- for development purposes (with auto reload) first run `SET FLASK_ENV=development` and then `flask run`
- in production `flask run`

# Team

* Miloš Radojčin, R2 15/2020 (@milosradojcin)
* Dejan Dokić, R2 17/2020 (@dejandokic)
* Luka Marić, R2 34/2020 (@LordElemelon)
