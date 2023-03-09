import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sklearn
#from prediction import predict
#st.write("""
st.title (':blue [Heart disease Prediction App]')
st.write("""-- This app predicts A patient has a heart disease or not --

""")

st.sidebar.header('Please Input Features Value')

# Collects user input features into dataframe
#thall,cp,caa,thalachh,oldpeak,slp 
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


#thall,cp,caa,thalachh,oldpeak,slp 
    data = {'age':age, 'sex':sex, 'cp':cp, 'trtbps':trtbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalachh':thalachh,
       'exng':exng, 'oldpeak':oldpeak, 'slp':slp, 'caa':caa, 'thall':thall
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
#st.button("Predict type of Disease")
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
    clf = joblib.load("model_LogR.sav")
    return clf.predict(data)


# Apply model to make predictions
#prediction = predict(input_df)
#prediction_proba = model.predict_proba(input_df)
if st.button("Click here to Predict type of Disease"):
    result = predict(input_df)

    if (result[0]== 0):
        st.subheader('The Person :green[does not have a Heart Disease] :sunglasses:')
    else:
        st.subheader('The Person :red[has Heart Disease] :sad:')
        

#st.subheader('Prediction')
#st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)
