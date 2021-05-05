from .model_manager import train
from .glove import calculate_documents_glove
from .io_manager import merge_all_sections

if __name__ == '__main__':
    print('Training started. This may take some time...\n')

    # Necessary to run only once
    merge_all_sections()

    # Training
    train(model='word2vec')
    train(model='doc2vec')
    train(model='fasttext')
    train(model='tfidf')
    calculate_documents_glove()

    print('Everything finished successfully. You are ready to use the application.')
