from flask import Flask, render_template, request
import joblib


app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('crop_yield_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    
    output = round(prediction[0], 2)
    
    return render_template('result.html', prediction=output)





    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    