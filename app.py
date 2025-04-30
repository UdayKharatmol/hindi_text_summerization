from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your .h5 or .keras model
model = load_model('HindiTextModel.h5')  # Replace with your model file
model.save('hindisummmodel.keras')  # Save in the newer .keras format

# Define a preprocessing function if needed
def preprocess_text(text):
    # Example dummy preprocessing – replace with your actual logic
    # Convert text to input format required by your model
    # Example: tokenizer.texts_to_sequences, padding, etc.
    # For now, we use dummy input
    return np.array([[len(text) % 10]])  # placeholder input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    processed_input = preprocess_text(text)

    prediction = model.predict(processed_input)
    predicted_value = prediction[0].tolist()  # Or convert to string/int if needed

    return jsonify({'input': text, 'prediction': predicted_value})

if __name__ == '__main__':
    app.run(debug=True)
