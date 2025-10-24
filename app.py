from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load XGBoost model
model = pickle.load(open('xgb.pkl', 'rb'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    query = request.args
    try:
        inputs = [float(query.get(f'input_{i}')) for i in range(1, 9)]
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid or missing input parameters'}), 400

    pred_name = model.predict([inputs]).tolist()[0]
    return jsonify({'prediction': pred_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
