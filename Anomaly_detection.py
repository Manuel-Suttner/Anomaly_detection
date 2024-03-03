import sys
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import time

app = Flask(__name__)

# Define parameters
num_samples = 1
time_interval = 1  # in seconds
data_rate = 1 / time_interval  # data rate per second

# Define parameters for generating synthetic sensor data
temperature_mean = 25
temperature_std = 5
humidity_mean = 50
humidity_std = 10
sound_volume_mean = 60
sound_volume_std = 20

# Initialize counters for anomalies and total data points processed
total_anomalies_detected = 0
total_samples_processed = 0

# Generate timestamps
timestamps = pd.date_range(start=pd.Timestamp.now(), periods=num_samples,
                           freq=f'{time_interval}s')

# List to store detected anomalies
detected_anomalies = []


# Generate synthetic sensor data with more realistic patterns
def generate_sensor_data(num_samples):
    timestamp = pd.date_range(start=pd.Timestamp.now(), periods=num_samples, freq=f'{time_interval}s')
    temperature_mean = 25
    temperature_seasonal_amp = 5  # Amplitude of seasonal variation
    temperature_daily_period = 24  # Period of daily fluctuations (in hours)
    temperature_daily_amp = 2  # Amplitude of daily fluctuations
    temperature_noise_std = 1  # Standard deviation of random noise

    humidity_mean = 50
    humidity_seasonal_amp = 10
    humidity_daily_period = 24
    humidity_daily_amp = 5
    humidity_noise_std = 2

    sound_volume_mean = 60
    sound_volume_seasonal_amp = 20
    sound_volume_daily_period = 24
    sound_volume_daily_amp = 10
    sound_volume_noise_std = 5

    temperature = (temperature_mean +
                   temperature_seasonal_amp * np.sin(2 * np.pi * timestamp.dayofyear / 365) +
                   temperature_daily_amp * np.sin(2 * np.pi * timestamp.hour / temperature_daily_period) +
                   np.random.normal(scale=temperature_noise_std, size=num_samples))

    humidity = (humidity_mean +
                humidity_seasonal_amp * np.sin(2 * np.pi * timestamp.dayofyear / 365) +
                humidity_daily_amp * np.sin(2 * np.pi * timestamp.hour / humidity_daily_period) +
                np.random.normal(scale=humidity_noise_std, size=num_samples))

    sound_volume = (sound_volume_mean +
                    sound_volume_seasonal_amp * np.sin(2 * np.pi * timestamp.dayofyear / 365) +
                    sound_volume_daily_amp * np.sin(2 * np.pi * timestamp.hour / sound_volume_daily_period) +
                    np.random.normal(scale=sound_volume_noise_std, size=num_samples))

    sensor_data = pd.DataFrame({
        'Timestamp': timestamp,
        'Temperature': temperature,
        'Humidity': humidity,
        'SoundVolume': sound_volume
    })

    return sensor_data


sensor_data = generate_sensor_data(num_samples)

# Define parameters for anomaly detection
window_size = 50  # Size of the rolling window for calculating statistics
std_threshold = 3  # Number of standard deviations away from the mean to consider as anomaly


# Detect anomalies for each sensor type
def detect_anomalies(sensor_data, sensor_type):
    rolling_mean = sensor_data[sensor_type].rolling(window=window_size).mean()
    rolling_std = sensor_data[sensor_type].rolling(window=window_size).std()
    upper_bound = rolling_mean + std_threshold * rolling_std
    lower_bound = rolling_mean - std_threshold * rolling_std
    anomalies = sensor_data[(sensor_data[sensor_type] > upper_bound) | (sensor_data[sensor_type] < lower_bound)]
    return anomalies


@app.route('/predict', methods=['POST'])
def predict():
    global total_samples_processed, total_anomalies_detected

    try:
        # Validate input data
        data = request.get_json()
        if not isinstance(data,
                          dict) or 'Temperature' not in data or 'Humidity' not in data or 'SoundVolume' not in data:
            raise ValueError("Invalid input format. Please provide Temperature, Humidity, and SoundVolume data.")

        # Receive sensor data
        temperature_data = pd.DataFrame(data['Temperature'])
        humidity_data = pd.DataFrame(data['Humidity'])
        sound_volume_data = pd.DataFrame(data['SoundVolume'])

        # Detect anomalies for each sensor type
        temperature_anomalies = detect_anomalies(temperature_data, 'Temperature')
        humidity_anomalies = detect_anomalies(humidity_data, 'Humidity')
        sound_volume_anomalies = detect_anomalies(sound_volume_data, 'SoundVolume')

        # Update model performance metrics
        total_samples_processed += 1
        total_anomalies_detected += len(temperature_anomalies) + len(humidity_anomalies) + len(sound_volume_anomalies)

        # Prepare response with timestamps and confidence scores
        response = {
            'Temperature': [{
                'Timestamp': str(row['Timestamp']),
                'ConfidenceScore': (row['Temperature'] - temperature_data['Temperature'].mean()) / temperature_data[
                    'Temperature'].std()
            } for _, row in temperature_anomalies.iterrows()],
            'Humidity': [{
                'Timestamp': str(row['Timestamp']),
                'ConfidenceScore': (row['Humidity'] - humidity_data['Humidity'].mean()) / humidity_data[
                    'Humidity'].std()
            } for _, row in humidity_anomalies.iterrows()],
            'SoundVolume': [{
                'Timestamp': str(row['Timestamp']),
                'ConfidenceScore': (row['SoundVolume'] - sound_volume_data['SoundVolume'].mean()) / sound_volume_data[
                    'SoundVolume'].std()
            } for _, row in sound_volume_anomalies.iterrows()]
        }

        return jsonify(response), 200

    except Exception as e:
        error_message = "An error occurred while processing the request: " + str(e)
        return jsonify({'error': error_message}), 500


anomaly_percentage = 0  # Initialize anomaly_percentage as a global variable
# Detect anomalies for each sensor type
temperature_anomalies = detect_anomalies(sensor_data, 'Temperature')
humidity_anomalies = detect_anomalies(sensor_data, 'Humidity')
sound_volume_anomalies = detect_anomalies(sensor_data, 'SoundVolume')


@app.route('/model_performance', methods=['GET'])
def get_model_performance():
    global total_samples_processed, total_anomalies_detected

    anomaly_percentage = (total_anomalies_detected / total_samples_processed) * 100 \
        if total_samples_processed > 0 else 0
    performance = {
        'TotalSamplesProcessed': total_samples_processed,
        'TotalAnomaliesDetected': total_anomalies_detected,
        'AnomalyPercentage': anomaly_percentage
    }
    return jsonify(performance), 200


# Simulate continuous data streaming
try:
    while True:
        try:
            # Generate new data point
            new_data_point = {
                'Timestamp': pd.Timestamp.now(),
                'Temperature': np.random.normal(loc=temperature_mean, scale=temperature_std),
                'Humidity': np.random.normal(loc=humidity_mean, scale=humidity_std),
                'SoundVolume': np.random.normal(loc=sound_volume_mean, scale=sound_volume_std)
            }
            # Concatenate new data point to DataFrame
            sensor_data = pd.concat([sensor_data, pd.DataFrame(new_data_point, index=[0])], ignore_index=True)
            # Print the latest data point
            print(sensor_data.tail(1))  # Print the last row (latest data point)

            # Combine anomalies from all sensor types
            all_anomalies = pd.concat([temperature_anomalies, humidity_anomalies,
                                       sound_volume_anomalies]).drop_duplicates()

            total_samples_processed += 1
            total_anomalies_detected += len(all_anomalies)

            # Delay to ensure one data point generated per second
            time.sleep(time_interval)

        except KeyboardInterrupt:
            raise

except KeyboardInterrupt:
    print("Streaming interrupted. Exiting gracefully...")

# Print detected anomalies after data generation stops
print("Total Samples Processed:", total_samples_processed)
print("Total Anomalies Detected:", total_anomalies_detected)
print("Temperature anomalies detected:", len(temperature_anomalies))
print("Humidity anomalies detected:", len(humidity_anomalies))
print("Sound volume anomalies detected:", len(sound_volume_anomalies))
