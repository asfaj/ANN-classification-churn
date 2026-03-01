import streamlit as st 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

## Load the trained model 
model=tf.keras.models.load_model('model.h5')

## load the encoder and scaler 
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender= st.selectbox('Gender', label_encoder_gender.classes_)
age= st.slider('Age', 18, 92)
balance= st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary= st.number_input('estimated salary')
tenure=st.slider('Tenure', 0,10)
num_of_products=st.slider('Number of products', 1, 14)
has_cr_card=st.selectbox('Has credit card', [0,1])
is_active_member= st.selectbox('is active member', [0,1])

input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# Encode Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# One-hot encode Geography
geo_encoded= onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(
    geo_encoded, columns=onehot_encoder_geography.get_feature_names_out()
)

# Combine and drop original Geography column
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data = input_data.drop('Geography', axis=1)

# Reorder columns to match scaler's expected feature order
expected_columns = scaler.get_feature_names_out()
input_data = input_data[expected_columns]

input_data_scaled=scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")