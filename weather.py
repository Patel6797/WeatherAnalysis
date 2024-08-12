import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to fetch weather data
def fetch_weather_data(api_key, lat, lon, date):
    url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {
        'lat': lat,
        'lon': lon,
        'dt': int(pd.Timestamp(date).timestamp()),
        'appid': api_key,
        'units': 'metric'
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to process weather data
def process_weather_data(raw_data):
    weather_data = []
    for entry in raw_data['hourly']:
        weather_data.append({
            'datetime': pd.to_datetime(entry['dt'], unit='s'),
            'temperature': entry['temp'],
            'humidity': entry['humidity'],
            'weather': entry['weather'][0]['description'],
        })
    return pd.DataFrame(weather_data)

# Function to calculate daily averages
def calculate_daily_averages(df):
    return df.resample('D', on='datetime').mean()

# Function to plot weather data
def plot_weather_data(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='datetime', y='temperature', data=df)
    plt.title('Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.show()

# Function to apply linear regression for temperature prediction
def linear_regression_model(df):
    df['day_of_year'] = df['datetime'].dt.dayofyear
    X = df[['day_of_year']]
    y = df['temperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, predictions, color='red')
    plt.title('Temperature Prediction')
    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (°C)')
    plt.show()

# Main function to run the project
def main():
    api_key = "cc31645a146f69d1e1ec4d9e75505993" # Replace with your OpenWeatherMap API key
    lat = 51.626186  # Latitude for location (e.g., Tokyo)
    lon = -0.389276  # Longitude for location (e.g., Tokyo)
    dates = pd.date_range(start='2023-01-01', end='2023-01-07')  # Date range

    all_weather_data = pd.DataFrame()

    for date in dates:
        raw_data = fetch_weather_data(api_key, lat, lon, date)
        daily_data = process_weather_data(raw_data)
        all_weather_data = pd.concat([all_weather_data, daily_data])

    all_weather_data = calculate_daily_averages(all_weather_data)

    plot_weather_data(all_weather_data)
    linear_regression_model(all_weather_data)

if __name__ == "__main__":
    main()