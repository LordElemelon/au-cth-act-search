from .model_manager import train
from .glove import calculate_documents_glove
from .io_manager import merge_all_sections
from time import perf_counter

if __name__ == '__main__':
    start = perf_counter()
    print("Starting!")

    # Necessary to run only once
    # merge_all_sections()
    # print("merging sections after %.2f sec, %.2f min" % ((perf_counter() - start), (perf_counter() - start) / 60))

    # Training
    # train(model='word2vec')
    # print("word2vec trained after %.2f sec, %.2f min" % ((perf_counter() - start), (perf_counter() - start) / 60))
    # train(model='doc2vec')
    # print("doc2vec trained after %.2f sec, %.2f min" % ((perf_counter() - start), (perf_counter() - start) / 60))
    # train(model='fasttext')
    # print("fasttext trained after %.2f sec, %.2f min" % ((perf_counter() - start), (perf_counter() - start) / 60))
    # calculate_documents_glove()

    end = perf_counter()
    print("Took time: %.2f sec, %.2f min" % ((end - start), (end - start) / 60))
