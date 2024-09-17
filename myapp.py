import streamlit as st
 
st.title('My First Streamlit App')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title of the app
st.title('Weather Prediction with Multiple Variables')

# Load data
@st.cache
def load_data():
    data = pd.read_csv('weather.csv')
    # Clean column names
    data.columns = data.columns.str.strip()
    return data

data = load_data()

# Display column names to check
st.write("Columns in the dataset:", data.columns)

# Prepare the features and target variable
features = ['humidity', 'wind_speed', 'precipitation']
X = data[features]
y = data['temperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model parameters
st.write(f"Intercept: {model.intercept_}")
st.write(f"Coefficients: {model.coef_}")

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Visualize predictions vs true values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel('True Values')
ax.set_ylabel('Predictions')
ax.set_title('Predictions vs True Values')
st.pyplot(fig)

# User input for prediction
st.sidebar.header('User Input')
humidity_input = st.sidebar.slider('Select Humidity (%)', min_value=0, max_value=100, value=50)
wind_speed_input = st.sidebar.slider('Select Wind Speed (km/h)', min_value=0, max_value=50, value=10)
precipitation_input = st.sidebar.slider('Select Precipitation (cm)', min_value=0.0, max_value=5.0, value=0.2)

# Make prediction based on user input
input_data = np.array([[humidity_input, wind_speed_input, precipitation_input]])
temperature_pred = model.predict(input_data)
st.sidebar.write(f'Predicted Temperature: {temperature_pred[0]:.2f}°C')
