from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
# Define forecast mapping for energy output calculation
forecast_mapping = {
    'Berjerebu': 0.1,  # Hazy
    'Tiada hujan': 0.2,  # No rain
    'Hujan': 0.3,  # Rain
    'Hujan di beberapa tempat': 0.4,  # Scattered rain
    'Hujan di satu dua tempat': 0.5,  # Isolated rain
    'Hujan di satu dua tempat di kawasan pantai': 0.5,  # Isolated rain over coastal areas
    'Hujan di satu dua tempat di kawasan pedalaman': 0.5,  # Isolated rain over inland areas
    'Ribut petir': 0.6,  # Thunderstorms
    'Ribut petir di beberapa tempat': 0.7,  # Scattered thunderstorms
    'Ribut petir di beberapa tempat di kawasan pedalaman': 0.7,  # Scattered thunderstorms over inland areas
    'Ribut petir di satu dua tempat': 0.8,  # Isolated thunderstorms
    'Ribut petir di satu dua tempat di kawasan pantai': 0.8,  # Isolated thunderstorms over coastal areas
    'Ribut petir di satu dua tempat di kawasan pedalaman': 0.8  # Isolated thunderstorms over inland areas
}

# Load the trained models
with open('energy_pricing_model.pkl', 'rb') as file:
    energy_pricing_model = pickle.load(file)

with open('energydemand_model.pkl', 'rb') as pkl_file:
    energy_demand_model = pickle.load(pkl_file)



# Define input shape for the LSTM model
n_timesteps = 1  # Adjust based on your model's configuration
n_features = 6   # Adjust based on your model's configuration

@app.route('/')
def home():
    return "Energy and Weather Forecasting API"

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()
    features = [data['current_supply'], data['current_demand'], data['hour'], data['day_of_week'], data['historical_price']]
    prediction = energy_pricing_model.predict([features])
    return jsonify({'prediction': prediction[0]})

@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.get_json()
    periods = data.get('periods', 10)  # Number of periods to forecast, default to 10

    # Make predictions
    forecast = energy_demand_model.forecast(steps=periods)
    forecast = forecast.tolist()  # Convert forecast to list

    return jsonify({'forecast': forecast})

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    # Fetch weather forecast data from the API
    response = requests.get('https://api.data.gov.my/weather/forecast/')

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()

        # Create a list to hold the processed data
        records = []

        # Process each item in the JSON data
        for item in data:
            record = {
                'Location': item['location']['location_name'],
                'Date': item['date'],
                'Morning Forecast': item['morning_forecast'],
                'Afternoon Forecast': item['afternoon_forecast'],
                'Night Forecast': item['night_forecast'],
                'Summary Forecast': item['summary_forecast'],
                'Summary When': item['summary_when'],
                'Min Temperature (°C)': item['min_temp'],
                'Max Temperature (°C)': item['max_temp']
            }
            
            # Calculate energy output for each record
            record['Energy Output'] = record['Max Temperature (°C)'] * 0.5 + forecast_mapping.get(record['Morning Forecast'], 0) * 100
            
            records.append(record)

        # Return the energy output data as JSON
        return jsonify(records)
    
    else:
        # Return an error message if the API request fails
        return jsonify({'error': f'Unable to fetch data. Status code: {response.status_code}'}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
