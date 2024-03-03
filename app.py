import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer, LSTM, Conv1D, Input, Dense, Dropout, MaxPooling1D, Flatten, concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.utils import plot_model
from datetime import timedelta
import requests
from bs4 import BeautifulSoup


import os
import streamlit as st
import requests
import base64

st.set_page_config(
    layout="wide",
)



@st.cache_data
def get_img_as_base64(url):
    response = requests.get(url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode()
    return None

# New image URL for the sidebar
background_img_url = "https://images.pexels.com/photos/15848930/pexels-photo-15848930/free-photo-of-yellow-gradient-background.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
background_img = get_img_as_base64(background_img_url)

# Sidebar styling with the new image as background and opacity
sidebar_bg_img = f"""
<style>
[data-testid="stSidebar"] {{
    position: relative;
    background-image: url('data:image/jpeg;base64,{background_img}');
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    color: white;
    padding: 20px;
    border-radius: 5px; 
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}


</style>
"""

# Apply the custom styling to the sidebar
st.sidebar.markdown(sidebar_bg_img, unsafe_allow_html=True)

# Main content styling
st.markdown(
    """
    <style>
        .main {
            background-image: url("https://images.unsplash.com/photo-1492648841336-8e8503376b73?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-position: top left;
            background-repeat: no-repeat;
            background-attachment: local;
            padding: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Main content styling
st.markdown(
    """
    <style>
        .main {
            background-image: url("https://images.unsplash.com/photo-1492648841336-8e8503376b73?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-position: top left;
            background-repeat: no-repeat;
            background-attachment: local;
            padding: 20px;
            background-size: cover;
        }
        .sidebar {
            background: #2E3B4E; /* Dark background color */
            color: white;
            padding: 20px;
            border-radius: 10px; /* Rounded corners */
        }
    </style>
    """,
    unsafe_allow_html=True
)


# def scrapegold_prices():
#     url = "https://www.fenegosida.org/"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, "html.parser")
#     gold_prices = []
#     for div in soup.find_all("div", class_="rate-gold"):
#         price = div.find("div", class_="rate-gold post").text.strip()
#         gold_prices.append(price)
#     return gold_prices

# # Call the function and print the result
# gold_prices = scrapegold_prices()
# for price in gold_prices:
#     print(price)



# Custom LSTM layer
class CustomLSTM(Layer):
    def __init__(self, units, return_sequences=False, **kwargs): 
        super(CustomLSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequences_ = return_sequences

    def build(self, input_shape):
        self.lstm_layer = LSTM(
            units=self.units,
            return_sequences=self.return_sequences_,
        )
        super(CustomLSTM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.lstm_layer(inputs, **kwargs)

    def get_config(self):
        config = super(CustomLSTM, self).get_config()
        config.update({"units": self.units, "return_sequences": self.return_sequences_})
        return config

# Custom CNN layer
class CustomCNN(Layer):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super(CustomCNN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        self.conv1d_layer = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
        )
        super(CustomCNN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.conv1d_layer(inputs, **kwargs)

    def get_config(self):
        config = super(CustomCNN, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size, "activation": self.activation})
        return config

# Load the preprocessed dataset
dataset = pd.read_csv('gold_data_3.csv')

# Convert the 'Date' column to datetime format
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%m/%d/%Y')
dataset.dropna(subset=['Vol.'], inplace=True)

# Define a function to convert a single value from string to float and remove non-numeric characters
def convert_to_float_and_remove_chars(value):
    try:
        # Remove non-numeric characters, such as ',' and '%' or any other units
        value = value.replace(',', '').replace('%', '').replace('K', '')
        return float(value)
    except ValueError:
        return np.nan
    
def preprocess_user_date(user_date):
    if isinstance(user_date, date):
        user_date = datetime.combine(user_date, datetime.min.time())  # Convert to datetime
    user_date_numeric = np.array([user_date.timestamp()])
    return user_date_numeric

# Function to preprocess user input numerical features
def preprocess_user_numerical_input(x_numerical_reshaped):
    
    user_num_input = np.zeros((1, x_numerical_reshaped.shape[1], 1))
    user_num_input[0] = sc_x.transform(np.zeros((1, x_numerical_reshaped.shape[1]))).reshape(x_numerical_reshaped.shape[1], 1)
    return user_num_input

# Apply the conversion function to the relevant columns in the dataset
columns_to_convert = ['Open', 'High', 'Low', 'Vol.', 'Change %', 'Price']

for column in columns_to_convert:
    dataset[column] = dataset[column].apply(convert_to_float_and_remove_chars)

# Drop rows with missing values (NaN)
dataset = dataset.dropna()

# Select relevant columns for input features
x_numerical = dataset[['Open', 'High', 'Low', 'Vol.', 'Change %']].values
x_date = dataset['Date'].values
y = dataset['Price'].values

# Convert numpy datetime64 to Python datetime objects
x_date = np.array([pd.Timestamp(date).to_pydatetime() for date in x_date])

# Convert datetime to numerical values
x_date_numeric = np.array([date.timestamp() for date in x_date])

# Split the data into training and testing sets
x_num_train, x_num_test, x_date_train, x_date_test, y_train, y_test = train_test_split(
    x_numerical, x_date_numeric, y, test_size=0.2, random_state=0
)


# Standardize the numerical input features
sc_x = StandardScaler()
x_numerical_scaled = sc_x.fit_transform(x_numerical)

# Reshape the numerical input data for LSTM
x_numerical_reshaped = x_numerical_scaled.reshape(x_numerical_scaled.shape[0], x_numerical_scaled.shape[1], 1)

# Create the numerical feature model
numerical_input = Input(shape=(x_numerical_reshaped.shape[1], 1))
lstm_layer_num = CustomLSTM(100, return_sequences=True)(numerical_input)
dropout_num = Dropout(0.5)(lstm_layer_num)
cnn_layer_num = CustomCNN(64, kernel_size=3, activation='relu')(dropout_num)
pooling_num = MaxPooling1D(pool_size=2)(cnn_layer_num)
flatten_num = Flatten()(pooling_num)
dropout_num_2 = Dropout(0.5)(flatten_num)

# Create the date feature model
date_input = Input(shape=(1,))
dense_date = Dense(20, activation='relu')(date_input)

# Concatenate the output of the numerical model and the date model
concatenated_features = concatenate([dropout_num_2, dense_date])

# Dense layer for final prediction
output = Dense(1)(concatenated_features)

# Create the model
model = load_model('modelv3.h5', custom_objects={'CustomLSTM': CustomLSTM, 'CustomCNN': CustomCNN})

# Standardize the numerical input features
x_num_train_scaled = sc_x.fit_transform(x_num_train)
x_num_test_scaled = sc_x.transform(x_num_test)

# Reshape the numerical input data for LSTM
x_num_train_reshaped = x_num_train_scaled.reshape(x_num_train_scaled.shape[0], x_num_train_scaled.shape[1], 1)
x_num_test_reshaped = x_num_test_scaled.reshape(x_num_test_scaled.shape[0], x_num_test_scaled.shape[1], 1)

# Function to get an image in base64 format from a local file
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Image path for the title background
img_path = "bgv2.png"  # Replace with the path to your local image

# Get the base64-encoded image
img = get_img_as_base64(img_path)

# HTML styling for the title with background image
title_bg_img = f"""
<div style="background-image: url('data:image/jpeg;base64,{img}');
            background-size: cover;
            background-position: center;
            padding: 70px ;
            text-align: cover;
            font-size: 36px;
            border-radius: 5px;
            margin: 0;
            overflow: hidden;">
    Gold Price Prediction
</div>
"""
# Apply the custom styling to the title
st.markdown(title_bg_img, unsafe_allow_html=True)

logo_path = "4.png"
st.sidebar.image(logo_path,use_column_width=True)

st.sidebar.write(' ')

# Display the h3 heading with increased font size
st.sidebar.markdown('<h3 style="font-size: 18px;">Select a date:</h3>', unsafe_allow_html=True)

# Get the current date
current_date = datetime.now()

max_date = current_date + timedelta(days=365)

def scrapegold_prices():
    url = "https://www.fenegosida.org/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    gold_prices = []
    for div in soup.find_all("div", class_="rate-gold post"):
        price = div.text.strip()
        gold_prices.append(price)
    return gold_prices
import base64

def gif_to_base64(gif_path):
    with open(gif_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return base64_image

# Path to your GIF file
gif_path = "goldp.gif"

# Convert GIF to Base64
base64_image = gif_to_base64(gif_path)

# Call the function to scrape gold prices
gold_prices = scrapegold_prices()

# Display the gold prices on Streamlit frontend
st.title("Today's Actual Gold Price")

# Display the gold prices and GIFs using the same container
for price in gold_prices:
    if "FINE GOLD (9999)" in price:
        # Remove the unwanted characters from the price
        cleaned_price = price.replace("ЯЦЂ", "").replace("Яц░"," ").replace("Nrs", " Nrs").replace("(9999)", " ").replace("1 tola", "1 tola Nrs").replace("121000", "121000/-")
        
        # Display the GIF and gold price in the same container
        st.markdown(
            f"""
            <style>
                .price-and-gif-container {{
                    display: flex;
                    flex-direction: row;
                    align-items: center;
                    justify-content: center;
                    background-color: rgba(128, 128, 128, 0.3);
                    padding: 10px;
                    border-radius: 10px;
                    margin: 10px;
                    font-size: 25px;
                }}

                .price-and-gif-container img {{
                    width: 100px;
                    margin-right: 20px;
                    
                }}
            </style>
            <div class="price-and-gif-container">
                <img src="data:image/gif;base64,{base64_image}">
                <div style="text-align: center;">
                    {cleaned_price}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# Display the date_input without a separate label

# Get the current date
current_date = datetime.now()

# Limit the date input to today and one year from today
max_date = current_date + timedelta(days=365)
min_date = current_date

# Display the date input widget with appropriate limits
user_date = st.sidebar.date_input("", key="user_date", min_value=min_date, max_value=max_date)
user_date_numeric = preprocess_user_date(user_date)
user_num_input = preprocess_user_numerical_input(x_numerical_reshaped)
# Prediction
prediction = model.predict([user_num_input, user_date_numeric])

# prediction is the predicted gold price for 1 troy ounce in dollars
prediction_in_dollars_per_troy_ounce = prediction[0][0]

# Conversion factors
dollars_to_nepali_rupees_exchange_rate = 133

# Convert from dollars to Nepali Rupees per tola
prediction_in_nepali_rupees = prediction_in_dollars_per_troy_ounce  * dollars_to_nepali_rupees_exchange_rate

prediction_in_nepali_rupees_per_tola = prediction_in_nepali_rupees * 0.373 
# prediction_in_nepali_rupees_per_tola_tax = prediction_in_nepali_rupees * 0.373 + (16.9/100 * prediction_in_nepali_rupees_per_tola)

prediction_in_nepali_rupees_per_tola_tax = prediction_in_nepali_rupees_per_tola + 21957





# Display predictions on the sidebar

st.sidebar.subheader(f"For 1 Tola Gold,")
st.sidebar.markdown(f'<h2 style="color: gold;">Predicted Gold Price (exclusive of tax):</h2>', unsafe_allow_html=True)
st.sidebar.subheader(f" NPR {prediction_in_nepali_rupees_per_tola:.2f}")

st.sidebar.markdown(f'<h2 style="color: gold;">Predicted Gold Price (inclusive of tax):</h2>', unsafe_allow_html=True)
st.sidebar.subheader(f" NPR {prediction_in_nepali_rupees_per_tola_tax:.2f}")

icon_path = "m.gif"
st.sidebar.image(icon_path, use_column_width=True)

# Display predictions on the sidebar
st.sidebar.subheader(f"For 1 Troy Ounce Gold,")
st.sidebar.markdown(f'<h2 style="color: gold;">Predicted Gold Price (in Dollar):</h2>', unsafe_allow_html=True)
st.sidebar.subheader(f" $ {prediction_in_dollars_per_troy_ounce:.2f}")
# st.sidebar.subheader(f" ")
st.sidebar.markdown(f'<h2 style="color: gold;">Predicted Gold Price (in Nepali Rupees):</h2>', unsafe_allow_html=True)

st.sidebar.subheader(f"NPR {prediction_in_nepali_rupees:.2f}")



st.write(' ')
st.write(' ')
st.write(' ')

# Plot the historical gold prices
st.subheader("Historical Gold Prices")
st.write(' ')
st.line_chart(dataset.set_index('Date')['Price'])

# Actual vs. Predicted Prices plot
st.header('Actual vs. Predicted Prices:')
st.write(' ')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test, label='Actual Prices', color='blue') 
ax.plot(model.predict([x_num_test_reshaped, x_date_test]), label='Predicted Prices', color='red') 
ax.set_xlabel('Number of Observations')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)


predictions = model.predict([x_num_test_reshaped, x_date_test])

# Calculate metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

st.sidebar.write(' ')
st.sidebar.write(' ')

# Create a red container with padding
container_style = "padding: 10px; background-color: rgba(0, 0, 0, 0.6); border-radius: 10px;"

# Calculate MAPE on training set
train_predictions = model.predict([x_num_train_reshaped, x_date_train])
predictions_test = model.predict([x_num_test_reshaped, x_date_test])
mape_train = np.mean(np.abs((y_train - train_predictions.flatten()) / y_train)) * 100

# Calculate MAPE on testing set
mape_test = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100

# Calculate RMAE and RMSE for training set
rmae_train = np.sqrt(mean_absolute_error(y_train, train_predictions.flatten()))
rmae_test = np.sqrt(mean_absolute_error(y_test, predictions.flatten()))


# Calculate RMSE for training set
rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions.flatten()))

# Calculate RMSE for testing set
rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test.flatten()))

# Scale RMSE values between 0 and 1 for better visualization
scaled_rmse_train = rmse_train / (np.max(y_train) - np.min(y_train))
scaled_rmse_test = rmse_test / (np.max(y_test) - np.min(y_test))

# Combine both sets of metrics in the same container
st.sidebar.markdown(
    f'<div style="{container_style}">'
    '<h3 style="color: gold;">Model Evaluation Metrics:</h3>'
    f'<p style="color: white;">Root Mean Absolute Error (RMAE) on Training Set: {rmae_train:.2f}</p>'
    f'<p style="color: white;">Root Mean Absolute Error (RMAE) on Testing Set: {rmae_test:.2f}</p>'
    f'<p style="color: white;">Scaled Root Mean Squared Error(RMSE) on Training Set: {scaled_rmse_train:.4f}</p>'
    f'<p style="color: white;">Scaled Root Mean Squared Error(RMSE) on Testing Set: {scaled_rmse_test:.4f}</p>'
    f'<p style="color: white;">Mean Absolute Percentage Error (MAPE) on Training Set: {mape_train:.2f}%</p>'
    f'<p style="color: white;">Mean Absolute Percentage Error (MAPE) on Testing Set: {mape_test:.2f}%</p>'
    '</div>',
    unsafe_allow_html=True
)


st.sidebar.write(' ')
st.sidebar.write(' ')

# Load the model with custom layers
loaded_model = load_model('modelv3.h5', custom_objects={'CustomLSTM': CustomLSTM, 'CustomCNN': CustomCNN})

# Display the model architecture
loaded_model.summary()

# Create a DataFrame with standardized numerical features
df_numerical = pd.DataFrame(x_numerical_scaled, columns=['Open', 'High', 'Low', 'Vol.', 'Change %'])

# Calculate correlation matrix
correlation_matrix = df_numerical.corr()

st.write(' ')
st.write(' ')

# Display the heatmap
st.subheader('Correlation Heatmap of Numerical Features:')
st.write(' ')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
st.pyplot(plt)

st.write(' ')

# Display the distribution of 'Price'
st.subheader('Distribution of Gold Prices:')
st.write(' ')
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.distplot(dataset['Price'],  color='blue')
plt.xlabel('Gold Price')
plt.ylabel('Density')
st.pyplot(plt)

