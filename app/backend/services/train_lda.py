import os
import re
import pickle
from time import perf_counter
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .config import Config
from .utils import preprocess

# import logging
# logging.basicConfig(filename='gensim.log',
#                     format="%(asctime)s:%(levelname)s:%(message)s",
#                     level=logging.INFO)


def jaccard_similarity(topic_1, topic_2):

    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection))/float(len(union))

if __name__ == '__main__':

    doc_names = os.listdir(Config.corpus_path)

    # Documents preprocess
    data_lemmatized = []
    for directory in doc_names:
        print(directory)
        dir_path = Config.corpus_path+"/"+directory
        dir_text = ""
        for sect_filename in os.listdir(dir_path):
            with open(dir_path+"/"+sect_filename) as sect_file:
                dir_text += sect_file.read().strip() + " "
        dir_text = re.sub(r'\.+', '.', dir_text)
        dir_text = re.sub(r'(\(|\))+', ' ', dir_text)
        dir_text = re.sub(r'\s+', ' ', dir_text)

        tokens = preprocess(dir_text, punct=True, stopwrd=True, lda_clear=True)
        data_lemmatized.append(tokens)

    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
                
    # Pickle dump
    with open(Config.lda_path+'/data.pickle', 'wb') as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(Config.lda_path+'/dict.pickle', 'wb') as handle:
        id2word.save(handle, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    with open(Config.lda_path+'/texts.pickle', 'wb') as handle:
        pickle.dump(texts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Pickle read
    # with open(Config.lda_path+'/data.pickle', 'rb') as handle:
    #     corpus = pickle.load(handle)
    # id2word = corpora.Dictionary.load(Config.lda_path+'/dict.pickle')
    # with open(Config.lda_path+'/texts.pickle', 'rb') as handle:
    #     texts = pickle.load(handle)
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    
    start = perf_counter()
    print("Starting!")
    # # Make LDA models
    # num_topics = list(range(43, 49, 1))
    # num_keywords = 50

    # LDA_models = {}
    # LDA_topics = {}
    # for i, num in enumerate(num_topics):
    #     LDA_models[i] = LdaModel(corpus=corpus,
    #                             id2word=id2word,
    #                             num_topics=num,
    #                             update_every=1,
    #                             chunksize=4000,
    #                             passes=20,
    #                             alpha='auto',
    #                             random_state=42)

    #     shown_topics = LDA_models[i].show_topics(num_topics=num, 
    #                                             num_words=num_keywords,
    #                                             formatted=False)
    #     LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
    #     print("Run for %d topics in %.2f mins" % (num, (perf_counter() - start) / 60))
    #     start = perf_counter()

    # # Calculate mean stabilities
    # LDA_stability = {}
    # for i in range(0, len(num_topics)-1):
    #     jaccard_sims = []
    #     for t1, topic1 in enumerate(LDA_topics[i]): # pylint: disable=unused-variable
    #         sims = []
    #         for t2, topic2 in enumerate(LDA_topics[i+1]): # pylint: disable=unused-variable
    #             sims.append(jaccard_similarity(topic1, topic2))    
            
    #         jaccard_sims.append(sims)    
        
    #     LDA_stability[num_topics[i]] = jaccard_sims
                    
    # mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]

    # # Make coherence models
    # coherences = []
    # for i in range(len(LDA_models))[:-1]:
    #     coherences.append(CoherenceModel(model=LDA_models[i], texts=texts, dictionary=id2word, coherence='c_v').get_coherence())
    #     print("Coherence model for %d created after %.2f mins" % (i, (perf_counter() - start) / 60))
    #     start = perf_counter()

    # # Get best model
    # coh_sta_diffs = [coherence - mean_stability for coherence, mean_stability in zip(coherences, mean_stabilities)] 
    # coh_sta_max = max(coh_sta_diffs)
    # coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
    # ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
    # ideal_topic_num = num_topics[ideal_topic_num_index]

    # print(ideal_topic_num)
    # print(coherences)
    # print(mean_stabilities)

    # print("Ready for graphing in %.2f mins" % ((perf_counter() - start) / 60))
    # # Graph values
    # plt.figure(figsize=(20,10))
    # ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
    # ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')

    # ax.axvline(x=ideal_topic_num, label='Ideal Number of Topics', color='black')
    # ax.axvspan(xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor='grey')

    # y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
    # ax.set_ylim([0, y_max])
    # ax.set_xlim([num_topics[0], num_topics[-1]-1])
                    
    # ax.axes.set_title('Model Metrics per Number of Topics', fontsize=25)
    # ax.set_ylabel('Metric Level', fontsize=20)
    # ax.set_xlabel('Number of Topics', fontsize=20)
    # plt.legend(fontsize=20)
    # plt.show() 

    # IDF for possible future
    # idf = {}
    # for doc in corpus:
    #     for word, freq in doc:
    #         if word not in idf:
    #             idf[word] = 0
    #         idf[word] = idf[word] + freq
    # print(sorted([(id2word[wid], fr) for wid, fr in idf.items()], key=lambda x: x[1], reverse=True))

    # FINAL LDA MODEL
    num_topics = 46
    num_keywords = 100
    lda_model = LdaModel(corpus=corpus,
                        id2word=id2word,
                        num_topics=num_topics,
                        update_every=1,
                        eval_every=1,
                        chunksize=4000,
                        passes=20,
                        iterations=100,
                        alpha='auto',
                        eta='auto',
                        random_state=42)
    
    print("Run for %d topics in %.2f mins" % (num_topics, (perf_counter() - start) / 60))
    start = perf_counter()
    # pprint(lda_model.print_topics(num_topics=num_topics, num_words=num_keywords))
    print("\n\n")

    topics_shown = lda_model.show_topics(num_topics=num_topics, num_words=num_keywords, formatted=False)
    # print(topics_shown)
    word_freq = {}
    word_sum = {}
    topics_to_save = {}
    for num, rep in topics_shown:
        if num not in topics_to_save:
            topics_to_save[num] = {}
        for word, freq in rep:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] = word_freq[word]+1
            if word not in word_sum:
                word_sum[word] = 0
            word_sum[word] = word_sum[word]+freq
            topics_to_save[num][word] = freq
    print(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    print("\n")
    print(sorted(word_sum.items(), key=lambda x: x[1], reverse=True))
    print("\n")
    for k, v in topics_to_save.items():
        print(k)
        for k2, v2 in v.items():
            print("\t", k2, v2)
    print("\n")

    total_belonging = {}
    for i, (doc_title, doc_bow) in enumerate(zip(doc_names, corpus)):
        belonging = lda_model[doc_bow]
        max_bel = max([x[1] for x in belonging])
        topics_bel = [x[0] for x in belonging if x[1] > 0.5*max_bel]
        for t_b in topics_bel:
            if t_b not in total_belonging:
                total_belonging[t_b] = []
            total_belonging[t_b].append(doc_title)
    # pprint(total_belonging)
    print("\n")
    print(sum([len(v) for v in total_belonging.values()]))

    with open(Config.lda_path+'/topics.pickle', 'wb') as handle:
        pickle.dump(topics_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(Config.lda_path+'/belong.pickle', 'wb') as handle:
        pickle.dump(total_belonging, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(lda_model[corpus[0]])
    # print("\n")
    # print(lda_model[corpus])

    # p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    # matches = [p.findall(l) for l in open('gensim.log')]
    # matches = [m for m in matches if len(m) > 0]
    # tuples = [t[0] for t in matches]
    # perplexity = [float(t[1]) for t in tuples]
    # liklihood = [float(t[0]) for t in tuples]
    # iter = list(range(0,len(tuples)*10,10))
    # plt.plot(iter,liklihood,c="black")
    # plt.ylabel("log liklihood")
    # plt.xlabel("iteration")
    # plt.title("Topic Model Convergence")
    # plt.grid()
    # plt.savefig("convergence_liklihood.pdf")
    # plt.close()


    print("Finished in %.2f mins" % ((perf_counter() - start) / 60))




