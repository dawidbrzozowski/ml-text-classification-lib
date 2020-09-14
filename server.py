from flask import Flask, request, jsonify

from predictors import Predictor
from utils.files_io import load_json

app = Flask(__name__)
predictor = Predictor(load_json('configs/data/predictor_config.json'))


@app.route('/predict', methods=['POST'])
def predict_api():
    req = request.get_json()
    data = req['data']
    return jsonify(predictor.predict(data)[0])


if __name__ == '__main__':
    app.run()
