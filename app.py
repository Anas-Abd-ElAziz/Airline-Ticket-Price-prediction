import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load('model.h5')

# Define the list of columns for input
input_cols = ['Airline', 'Source', 'Destination', 'Duration', 'Total_Stops',
              'Additional_Info', 'Journey_month', 'Weekday', 'Dep_hour', 'Dep_min']

# Define the list of airline options
airline_options = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
                   'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
                   'other', 'Jet Airways Business',
                   'Multiple carriers Premium economy']

# Define the list of source and destination options
source_options = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
destination_options = ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad']

# Define the list of Additional_Info options
additional_info_options = ['No info', 'In-flight meal not included', 'No check-in baggage included',
                           '1 Long layover', 'Change airports', 'Business class', 'other']

# Define the list of Total_Stops options
stops_options = [0, 1, 2, 3]

# Create a form to input the feature values
st.title("Flight Price Prediction")
with st.form(key='prediction_form'):
    inputs = []
    flight_date = st.date_input("Select Flight Date")
    departure_time = st.time_input("Select Departure Time")
    dep_hour = departure_time.hour if departure_time else None
    dep_min = departure_time.minute if departure_time else None
    duration_hours = st.slider("Duration (hours)", min_value=0, max_value=50, step=1)

    for col in input_cols:
        if col == 'Airline':
            value = st.selectbox(f"Select {col}", airline_options)
        elif col == 'Source':
            value = st.selectbox(f"Select {col}", source_options)
        elif col == 'Destination':
            value = st.selectbox(f"Select {col}", destination_options)
        elif col == 'Additional_Info':
            value = st.selectbox(f"Select {col}", additional_info_options)
        elif col == 'Journey_month':
            value = flight_date.month if flight_date else None
        elif col == 'Weekday':
            value = flight_date.weekday() if flight_date else None
        elif col == 'Dep_hour':
            value = dep_hour
        elif col == 'Dep_min':
            value = dep_min
        elif col == 'Total_Stops':
            value = st.selectbox(f"Select {col}", stops_options)
        elif col == 'Duration':
            value = duration_hours * 60  # Convert hours to minutes
        else:
            value = st.text_input(f"Enter {col}")
        inputs.append(value)
    submit_button = st.form_submit_button(label='Predict Price')

if submit_button:
    input_data = dict(zip(input_cols, inputs))
    input_df = pd.DataFrame([input_data])
    price_prediction = model.predict(input_df)

    # Display the predicted price
    st.header("Price Prediction")
    st.success(f"The predicted price is: {round(price_prediction[0])} $")
    