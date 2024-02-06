import streamlit as st
import pandas as pd
from joblib import load


model = load('house-data-lineer-model.pkl') 

st.title('House Price Estimate')
# Get input from user
floor = st.number_input('Floor', min_value=1, max_value=20, value=1)
number_rooms = st.number_input('Number of rooms', min_value=1, max_value=10, value=3)
size = st.number_input('Size of house', min_value=1, max_value=500, value=100)
address = st.selectbox('Address', ['Talas', 'Kocasinan', 'Melikgazi', 'Develi'])

# Convert the input to the format the model expects
features = pd.DataFrame([[number_rooms, floor, size, address]], columns=['Number Rooms', 'Floor', 'Size', 'Address'])

# Convert address with one-hot encoding
features = pd.get_dummies(features, columns=['Address'])

# Add columns for missing features and fill these columns with 0
missing_cols = set(['Address_Develi','Address_Kocasinan', 'Address_Melikgazi', 'Address_Talas']) - set(features.columns)
for c in missing_cols:
    features[c] = 0

# Determine the order of features the model expects
column_order = ['Number Rooms', 'Floor', 'Size','Address_Develi' , 'Address_Kocasinan', 'Address_Melikgazi', 'Address_Talas']

# Match the order of the features to the order the model expects
features = features[column_order]


# Run the model and get the prediction
prediction = model.predict(features)

# Show estimate to user
if prediction > 0:
    st.success(f'Estimated Property Value: {prediction[0]:,.2f} TL')
else:
    st.error(f'Estimated Property Value: {prediction[0]:,.2f} TL')


