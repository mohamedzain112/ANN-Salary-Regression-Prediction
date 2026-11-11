import streamlit as st
import pandas as pd 
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder 


#load the model
model = tf.keras.models.load_model('salary_regression_model.h5')

## Load the encoder and scaler
with open('onehotencoder_geography.pkl', 'rb') as f:
    geoencoder = pickle.load(f)

with open('labelencoder_gender.pkl', 'rb') as f:
    genderencoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)



## streamlit app
st.title("Salary Regression Prediction")

# User input
geography = st.selectbox('Geography', geoencoder.categories_[0])
gender = st.selectbox('Gender', genderencoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [genderencoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#one-hot encode 'Geography'
geo_encoder = geoencoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoder, columns=geoencoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale the input data
input_data = input_data[scaler.feature_names_in_]
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)   
prediction_salary = prediction[0][0]

st.write(f"The predicted salary is ${prediction_salary:.2f}.")