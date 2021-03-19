import torch
import pandas as pd

from backend.services import utils, io
from backend.services.config import Config
from backend.services.is_models import InferSent


def infersent(query, trained=True):
    V = 2
    MODEL_PATH = '../data/infersent/infersent_model.pkl' if trained else ('../data/infersent/infersent%s.pkl' % V)
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent_model = InferSent(params_model)
    infersent_model.load_state_dict(torch.load(MODEL_PATH))

    FASTTEXT_PATH = '../data/fasttext/crawl-300d-2M.vec'
    infersent_model.set_w2v_path(FASTTEXT_PATH)

    documents = [path_content[1] for path_content in io.read_documents_for_tfidf()]
    infersent_model.build_vocab([path_content[1] for path_content in documents], tokenize=True)

    if not trained:
        torch.save(infersent_model.state_dict(), '../data/infersent/infersent_model.pkl')

    query_vec = infersent_model.encode(query)[0]
    df = pd.DataFrame(columns=['doc', 'similarity'])
    for i, doc in enumerate(documents):
        sim = utils.cosine(query_vec, infersent_model.encode([doc[1]])[0])
        # print("Document = ", doc, "; similarity = ", sim)
        df.loc[i] = [doc[0], sim]

    df.sort_values(by=['similarity'], inplace=True, ignore_index=True, ascending=False)

    sections = []
    for i, row in df.iterrows():
        if i == Config.sections_to_display:
            break

        with open(df['doc'][i], 'r', encoding="utf8") as f:
            section = f.read()

            sections.append(section)
            print(section)

    return sections
