from sentence_transformers import SentenceTransformer
from . import utils, io_manager


def sentencebert(query):
    sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')  # try bert-base-nli-mean-tokens
    sentences = [path_content[1] for path_content in io_manager.read_documents_for_tfidf()]
    sentence_embeddings = sbert_model.encode(sentences)
    query_vec = sbert_model.encode([query])[0]

    for sent in sentences:
        sim = utils.cosine(query_vec, sbert_model.encode([sent])[0])
        print("Sentence = ", sent, "; similarity = ", sim)
