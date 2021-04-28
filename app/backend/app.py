from flask import Flask, request
from flask_cors import CORS, cross_origin
from services import main

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/basic-search', methods=['POST'])
@cross_origin()
def find_documents():
    return main.find_documents(request.json['query'], request.json['technique'])


@app.route('/qa', methods=['POST'])
@cross_origin()
def answer():
    return main.answer(request.json['question'], request.json['technique'])


@app.route('/read-sections', methods=['POST'])
@cross_origin()
def read_sections():
    return main.read_sections(request.json['names'])


if __name__ == '__main__':
    app.run(debug=True)
