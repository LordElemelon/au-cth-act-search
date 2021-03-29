from flask import Blueprint
from ..services import main


api = Blueprint('api', __name__)


@api.route('/word_embedding/<query>')
def find_documents(query):
    return main.find_documents(query, technique='fasttext')


@api.route('/qa/<question>')
def answer(question):
    return main.answer(question, technique='word2vec')
