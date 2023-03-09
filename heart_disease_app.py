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
st.text('The Person :green[does not have a Heart Disease] :sunglasses: 	:sparkling_heart:')
st.download_button('Download Sample file link for check', 'https://github.com/ripon2488/heart-disease-prediction-machine-learning/blob/main/heart_disease_dataset.csv')
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
st.write("""

# 
# 
### 🤠 About Me: 
#### 


- 🔭 I’m currently working on Database Administration, ML and DL.
- 🌱 I’m currently learning Data Analyics, Machine Learning ,Deep Learning and Data Science.
- 👯 I’m looking to collaborate on Data Engineering and AI.
- 🤔 I’m looking for help with Data Science and AI.
- 💬 Ask me about Data Analysis, Engineering and ML.
- 📫 How to reach me: <a href="https://www.linkedin.com/in/ripon2488/"> ripon2488 </a> 
- 📫 Workflow in Kaggle: <a href="https://www.kaggle.com/mdriponmiah"> mdriponmiah </a> 
- 😄 Pronouns: Ripon (রিপন)
- ⚡ Fun fact: 



#### More Details for me:  https://sites.google.com/view/ripon2488
### ✨ Tech Knowledge: 

![](https://img.shields.io/badge/python-3670A0?style=for-the-badge&amp;logo=python&amp;logoColor=ffdd54)
![](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&amp;logo=Keras&amp;logoColor=white)
![](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&amp;logo=numpy&amp;logoColor=white)
![](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&amp;logo=pandas&amp;logoColor=white)
![](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&amp;logo=plotly&amp;logoColor=white)
 ![](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&amp;logo=PyTorch&amp;logoColor=white)
![](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&amp;logo=scikit-learn&amp;logoColor=white) 
![](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&amp;logo=scipy&amp;logoColor=%white)
![](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&amp;logo=TensorFlow&amp;logoColor=white)
![](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&amp;logo=opencv&amp;logoColor=white)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F9de19bc8674de7e909cfdc555ab8199b%2Fpower%20bi.JPG?generation=1674674584825248&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F2984bf7961a04d79aa992de7e25fa036%2Ftableau.JPG?generation=1674674585096135&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F9b2e7383176bb5d806142e8ef8f89bb5%2Fgoogle%20data%20studio.JPG?generation=1674674585135850&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F224ba3f0a7a6dd52c6c5d57b4c6768bc%2Fmysql.JPG?generation=1674674585250106&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F451decb991b663ac2a01bf33287e8f89%2Foracle.JPG?generation=1674674585317290&alt=media)


![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmdriponmiah%2Fkaggle-badge&count_bg=%23DDAA17&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)


""")                                                                                  
