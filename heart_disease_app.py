import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sklearn
import joblib

st.image('header.png')
st.title(':blue[Heart disease Prediction App]')
st.write("""-- This app predicts A patient has a heart disease or not --

""")

st.sidebar.header('Please Input Features Value')

# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Age of persons: ')
    sex = st.sidebar.number_input('Gender of persons: ')
    cp = st.sidebar.selectbox('Chest pain type (4 values)',(0,1,2,3))
    trtbps = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestrol in mg/dl: ')
    fbs =  st.sidebar.number_input('Fasting blood sugar > 120 mg/dl: ')
    restecg = st.sidebar.selectbox('Resting electrocardio results:', ( 0,1,2))
    thalachh = st.sidebar.number_input('Maximum heart rate achieved thalach: ')
    exng = st.sidebar.number_input('Exercise induced angina: ')
    oldpeak = st.sidebar.number_input(' ST depression induced by exercise relative to rest (oldpeak): ')
    slp = st.sidebar.number_input('The slope of the peak exercise ST segment (slp): ')
    caa = st.sidebar.selectbox('Number of major vessels(0-3) colored by flourosopy (caa):',(0,1,2,3))
    thall = st.sidebar.selectbox(' Thall 0=normal, 1=fixed defect, 2 = reversable defect',(0,1,2))


    data = {'age':age, 'sex':sex, 'cp':cp, 'trtbps':trtbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalachh':thalachh,
       'exng':exng, 'oldpeak':oldpeak, 'slp':slp, 'caa':caa, 'thall':thall
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.write(input_df)

def predict(data):
    clf = joblib.load("model_LogR.sav")
    return clf.predict(data)


# Apply model to make predictions

if st.button("Click here to Predict type of Disease"):
    result = predict(input_df)

    if (result[0]== 0):
        st.subheader('The Person :green[does not have a Heart Disease] :sunglasses: 	:sparkling_heart:')
    else:
        st.subheader('The Person :red[has Heart Disease] :worried:')
        
#https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

