from allennlp.predictors import Predictor

from backend.services import main, model_manager


def allennlp(ques, embd_technique):
    text = ''
    if embd_technique == 'word2vec':
        word2vec_wv = model_manager.load_model(model='word2vec')

        text = ' '.join([s.strip() for s in main.find_documents_word2vec(ques, word2vec_wv)])
    elif embd_technique == 'doc2vec':
        docvec_model = model_manager.load_model(model='doc2vec')

        text = ' '.join([s.strip() for s in main.find_documents_doc2vec(ques, docvec_model)])
    elif embd_technique == 'fasttext':
        fasttext_wv = model_manager.load_model(model='fasttext')

        text = ' '.join([s.strip() for s in main.find_documents_fasttext(ques, fasttext_wv)])
    elif embd_technique == 'tfidf':
        text = ' '.join([s.strip() for s in main.find_documents_tfidf(ques)])
    elif embd_technique == 'glove':
        text = ' '.join([s.strip() for s in main.find_documents_glove(ques)])

    print('Text:', text)

    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz")
    prediction = predictor.predict(passage=text, question=ques)

    print('Answer:', prediction)
    return prediction
