# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model, label_encoder, symptom_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptom_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_symptoms = [int(request.form.get(symptom, 0)) for symptom in symptom_columns]
        X_input = np.array([input_symptoms])
        prediction = model.predict(X_input)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        return render_template('index.html', symptoms=symptom_columns, prediction_text=f'Predicted Disease: {predicted_disease}')
    except Exception as e:
        return render_template('index.html', symptoms=symptom_columns, prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
