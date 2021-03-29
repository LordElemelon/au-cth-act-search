from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from . import model_manager, main

import torch


# https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/
def bert(question, embd_technique):
    text = ''
    if embd_technique == 'word2vec':
        word2vec_wv = model_manager.load_model(model='word2vec')

        text = ' '.join([s.strip() for s in main.find_documents_word2vec(question, word2vec_wv)])
    elif embd_technique == 'doc2vec':
        docvec_model = model_manager.load_model(model='doc2vec')

        text = ' '.join([s.strip() for s in main.find_documents_doc2vec(question, docvec_model)])
    elif embd_technique == 'fasttext':
        fasttext_wv = model_manager.load_model(model='fasttext')

        text = ' '.join([s.strip() for s in main.find_documents_fasttext(question, fasttext_wv)])
    elif embd_technique == 'tfidf':
        text = ' '.join([s.strip() for s in main.find_documents_tfidf(question)])
    elif embd_technique == 'glove':
        text = ' '.join([s.strip() for s in main.find_documents_glove(question)])

    print('Text:', text)

    # Model
    # Also try: bert-large-uncased-whole-word-masking-finetuned-squad, deepset/roberta-base-squad2, distilbert-base-uncased-distilled-squad, distilbert-base-cased-distilled-squad
    # bert_model = BertForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2', return_dict = False) # , return_dict = False
    bert_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    # Tokenizer
    # Also try: bert-large-uncased-whole-word-masking-finetuned-squad, deepset/roberta-base-squad2, distilbert-base-uncased-distilled-squad, distilbert-base-cased-distilled-squad
    # tokenizer = BertTokenizer.from_pretrained('deepset/roberta-base-squad2')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

    # for section in read_documents_for_tfidf():
    input_ids = tokenizer.encode(question, text)

    # BERT only needs the token IDs, but for the purpose of inspecting the
    # tokenizer's behavior, let's also get the token strings and display them.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # Run our example through the model.
    start_scores, end_scores = bert_model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                          token_type_ids=torch.tensor([segment_ids]))  # The segment IDs to differentiate question from answer_text

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end + 1])

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer:', answer)

    return answer
