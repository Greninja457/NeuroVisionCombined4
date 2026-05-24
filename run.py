from flask import Flask, request, jsonify
from flask_cors import CORS

from app.classifier.route import classifier


app = Flask(__name__)
CORS(app)

app.register_blueprint(classifier, url_prefix='/classifier')


@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "NeuroVision API running", "image": None}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
