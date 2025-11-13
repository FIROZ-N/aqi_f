from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
def load_model_components():
    try:
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('feature_info.pkl', 'rb') as feature_file:
            feature_info = pickle.load(feature_file)
        
        return scaler, model, feature_info
    except Exception as e:
        print(f"Error loading model components: {e}")
        return None, None, None

scaler, model, feature_info = load_model_components()

# AQI Category function
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "green", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "#FFD700", "Air quality is acceptable. However, there may be a risk for some people."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "orange", "Members of sensitive groups may experience health effects."
    elif aqi_value <= 200:
        return "Unhealthy", "red", "Some members of the general public may experience health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "purple", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "#8B0000", "Health warning of emergency conditions."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict-aqi', methods=['POST'])
def predict_aqi():
    try:
        # Get form data
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no2 = float(request.form['no2'])
        co = float(request.form['co'])
        o3 = float(request.form['o3'])
        
        # Create input array
        input_data = np.array([[pm25, pm10, no2, co, o3]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get AQI category
        category, color, description = get_aqi_category(prediction)
        
        # Health recommendations based on AQI
        recommendations = get_health_recommendations(category)
        
        return render_template('result.html', 
                             prediction=round(prediction, 2),
                             category=category,
                             color=color,
                             description=description,
                             recommendations=recommendations,
                             input_data={
                                 'PM2.5': pm25,
                                 'PM10': pm10,
                                 'NO2': no2,
                                 'CO': co,
                                 'O3': o3
                             })
    
    except Exception as e:
        return render_template('predict.html', 
                             error=f"Error in prediction: {str(e)}")

def get_health_recommendations(category):
    recommendations = {
        "Good": [
            "Ideal air quality for outdoor activities",
            "Perfect day for exercise and outdoor events",
            "No restrictions needed"
        ],
        "Moderate": [
            "Unusually sensitive people should consider reducing prolonged outdoor exertion",
            "Generally acceptable for most activities",
            "Stay hydrated if exercising outdoors"
        ],
        "Unhealthy for Sensitive Groups": [
            "Sensitive groups should reduce outdoor activities",
            "People with heart or lung disease, older adults, and children should limit exertion",
            "Consider moving activities indoors"
        ],
        "Unhealthy": [
            "Everyone may begin to experience health effects",
            "Sensitive groups should avoid outdoor exertion",
            "General population should limit prolonged outdoor exercise"
        ],
        "Very Unhealthy": [
            "Health alert: Everyone may experience more serious health effects",
            "Avoid outdoor activities",
            "Keep windows closed and use air purifiers",
            "Sensitive groups should remain indoors"
        ],
        "Hazardous": [
            "Health warning of emergency conditions",
            "Everyone should avoid all physical activity outdoors",
            "Remain indoors and keep activity levels low",
            "Use high-efficiency air purifiers"
        ]
    }
    return recommendations.get(category, [])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['pm25', 'pm10', 'no2', 'co', 'o3']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create input array
        input_data = np.array([[data['pm25'], data['pm10'], data['no2'], 
                              data['co'], data['o3']]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get AQI category
        category, color, description = get_aqi_category(prediction)
        
        return jsonify({
            'aqi': round(prediction, 2),
            'category': category,
            'color': color,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)