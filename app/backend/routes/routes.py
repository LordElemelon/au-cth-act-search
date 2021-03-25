from flask import Blueprint, request
from services import main


api = Blueprint('api', __name__)


@api.route('/basic-search', methods=['POST'])
def find_documents():
    return main.find_documents(request.json['query'], technique='word2vec')


@api.route('/qa', methods=['POST'])
def answer():
    return main.answer(request.json['question'], technique='allennlp')
