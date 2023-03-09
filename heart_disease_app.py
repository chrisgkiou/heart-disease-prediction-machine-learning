import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from prediction import predict
st.write("""
# Heart disease Prediction App
This app predicts If a patient has a heart disease oe not

""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
#thall,cp,caa,thalachh,oldpeak,slp 
def user_input_features():
    
    thal = st.sidebar.selectbox('thal',(0,1,2))
    cp = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    ca = st.sidebar.selectbox('number of major vessels caa',(0,1,2,3))
    tha = st.sidebar.number_input('Maximum heart rate achieved thalachh: ')
    old = st.sidebar.number_input('oldpeak: ')
    slope = st.sidebar.number_input('he slope of the peak exercise ST segmen: ')


#thall,cp,caa,thalachh,oldpeak,slp 
    data = {'cp': cp,
            'thalachh':tha,
            'oldpeak':old,
            'slp':slope,
            'caa':ca,
            'thall':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
#heart_dataset = pd.read_csv('heart.csv')
#heart_dataset = heart_dataset.drop(columns=['target'])

#df = pd.concat([input_df,heart_dataset],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
#df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

#df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)
st.button(“Predict type of Disease”)
# Reads in saved classification model
#load_clf = pickle.load(open('model_svc.sav', 'rb'))
#-----------
#final_model = 'model_svc.sav'
#pickle.dump(final_model_svc, open(final_model, 'wb'))
 
# some time later...
 
# load the model from disk
#model = pickle.load(open(final_model, 'rb'))
#------------------

import joblib
def predict(data):
    clf = joblib.load("model_svc.sav")
    return clf.predict(data)


# Apply model to make predictions
#prediction = model.predict(input_df)
#prediction_proba = model.predict_proba(input_df)
if st.button(“Predict type of Disease”):
result = predict(input_df)
st.text(result[0])

#st.subheader('Prediction')
#st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)
