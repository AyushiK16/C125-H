from flask import Flask, jsonify, request
from classifier import getPrediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def addData():
    img = request.files.get('digit')
    #will search for the key 'digit' then pass in the prediction

    prediction = getPrediction(img)
    return jsonify({
        'prediction': prediction
    }),200

if __name__ == '__main__':
    app.run(debug=True)