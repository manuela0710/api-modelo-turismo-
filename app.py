
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)


model = joblib.load(os.path.join('model', 'tourist_model.pkl'))
model_columns = joblib.load(os.path.join('model', 'model_columns.pkl'))

@app.route('/')
def home():
    return {
        "message": "API para predecir tipo de turista (preferencias no verbales)",
        "instructions": "Envía un POST a /predict con JSON que contenga: age, GImg1, PImg1, Tense - relaxed, Hostile - friendly, sex_M"
    }

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return {
            "message": "¡Hola! Esta API predice el tipo de turista basado en señales no verbales.",
            "instructions": "Envía una solicitud POST a esta URL con un JSON que contenga: age, GImg1, PImg1, Tense - relaxed, Hostile - friendly, sex_M",
            "example": {
                "age": 35,
                "GImg1": 2,
                "PImg1": 2,
                "Tense - relaxed": 5,
                "Hostile - friendly": 6,
                "sex_M": 0
            }
        }
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Se requiere cuerpo JSON"}), 400

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities.max())

        return jsonify({
            "type_of_client": int(prediction),
            "confidence_percent": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
