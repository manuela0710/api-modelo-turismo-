# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Cargar modelo y columnas
model = joblib.load(os.path.join('model', 'tourist_model.pkl'))
model_columns = joblib.load(os.path.join('model', 'model_columns.pkl'))

@app.route('/')
def home():
    return {
        "message": "API para predecir tipo de turista (preferencias no verbales)",
        "instructions": "Envía un POST a /predict con JSON que contenga: age, GImg1, PImg1, Tense - relaxed, Hostile - friendly, sex_M"
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Se requiere cuerpo JSON"}), 400

        # Convertir a DataFrame y alinear columnas
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predecir
        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0].max())

        return jsonify({
            "type_of_client": pred,
            "confidence_percent": round(prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)