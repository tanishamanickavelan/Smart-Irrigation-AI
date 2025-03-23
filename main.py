def calculate_rainwater_collection(rainfall_mm, area_m2, efficiency=0.85):
    """
    Calculate rainwater collection.
    :param rainfall_mm: Rainfall in mm
    :param area_m2: Catchment area in square meters
    :param efficiency: Collection efficiency (default 85%)
    :return: Collected water in liters
    """
    volume_collected = rainfall_mm * area_m2 * efficiency  # 1mm rain = 1L/m²
    return round(volume_collected, 2)

if __name__ == "__main__":
    rainfall = float(input("Enter rainfall in mm: "))
    area = float(input("Enter catchment area in m²: "))
    print(f"Collected Rainwater: {calculate_rainwater_collection(rainfall, area)} liters")
def calculate_humidity_harvesting(humidity, air_volume, efficiency=0.4):
    """
    Estimate water collected from atmospheric humidity.
    :param humidity: Humidity percentage (0-100)
    :param air_volume: Volume of air processed in cubic meters
    :param efficiency: Conversion efficiency (default 40%)
    :return: Water collected in liters
    """
    max_water_content = air_volume * (humidity / 100) * 0.015  # 1m³ air ≈ 15g max water
    collected_water = max_water_content * efficiency
    return round(collected_water, 2)

if __name__ == "__main__":
    humidity = float(input("Enter humidity (%): "))
    air_volume = float(input("Enter air volume processed (m³): "))
    print(f"Collected Water: {calculate_humidity_harvesting(humidity, air_volume)} liters")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
def load_data():
    df = pd.read_csv("data/weather_data.csv")
    return df

# Train AI Model
def train_model():
    df = load_data()
    X = df[['temperature', 'humidity', 'soil_moisture']]
    y = df['water_required']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/water_model.pkl")
    print("AI Model Trained and Saved!")

# Predict Water Requirement
def predict_water(temperature, humidity, soil_moisture):
    model = joblib.load("models/water_model.pkl")
    prediction = model.predict([[temperature, humidity, soil_moisture]])
    return round(prediction[0], 2)

if __name__ == "__main__":
    train_model()
    temp = float(input("Enter temperature (°C): "))
    hum = float(input("Enter humidity (%): "))
    soil = float(input("Enter soil moisture (%): "))
    print(f"Predicted Water Requirement: {predict_water(temp, hum, soil)} liters")
from rainwater import calculate_rainwater_collection
from humidity import calculate_humidity_harvesting
from ai_water_opt import predict_water

def main():
    print("\n🌱 Sustainable Irrigation System 🌱")
    
    # Rainwater Harvesting
    rainfall = float(input("\nEnter Rainfall (mm): "))
    area = float(input("Enter Catchment Area (m²): "))
    rainwater = calculate_rainwater_collection(rainfall, area)
    print(f"Collected Rainwater: {rainwater} liters")
    
    # Atmospheric Moisture Harvesting
    humidity = float(input("\nEnter Humidity (%): "))
    air_volume = float(input("Enter Air Volume Processed (m³): "))
    humidity_water = calculate_humidity_harvesting(humidity, air_volume)
    print(f"Collected Atmospheric Water: {humidity_water} liters")

    # AI-based Water Requirement Prediction
    temp = float(input("\nEnter Temperature (°C): "))
    soil_moisture = float(input("Enter Soil Moisture (%): "))
    required_water = predict_water(temp, humidity, soil_moisture)
    print(f"💧 Predicted Water Requirement: {required_water} liters/day")

if __name__ == "__main__":
    main()
