from backend.services.model_manager import train
from backend.services.glove import calculate_documents_glove
from backend.services.io import merge_all_sections
from time import perf_counter

if __name__ == '__main__':
    start = perf_counter()

    # Training
    train(model='word2vec')
    train(model='doc2vec')
    train(model='fasttext')
    calculate_documents_glove()

    # Necessary to run only once
    merge_all_sections()

    end = perf_counter()
    print("Took time: %.2f sec, %.2f min" % ((end - start), (end - start) / 60))
