import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
import pickle
import pandas as pd


#Load the model
model=load_model('classification_notebooks\model.h5')

# load preprocessor objects
with open(file='classification_notebooks\label_encoder_gender.pkl',mode='rb') as file:
    label_encoder_obj=pickle.load(file=file)

with open(file='classification_notebooks\onehot_encoder_geo.pkl',mode='rb') as file:
    onehot_encoder_obj=pickle.load(file=file)   

with open(file='classification_notebooks\scaler.pkl',mode='rb') as file:
    scaler_obj=pickle.load(file=file)     

st.title('Customer Churn Prediction')    

#user input
geography=st.selectbox('Geography',onehot_encoder_obj.categories_[0])
gender=st.selectbox('Gender',label_encoder_obj.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_credit_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


input_data={
    'CreditScore': credit_score,
    'Geography':geography,
    'Gender':gender,
    'Age':age,
    'Tenure': tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_credit_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary
}


input_df=pd.DataFrame(data=[input_data])

#Gender
input_df['Gender']=label_encoder_obj.transform(input_df['Gender'])

#Geography
geo_encoded=onehot_encoder_obj.transform(input_df[['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(data=geo_encoded,
             columns=onehot_encoder_obj.get_feature_names_out(input_features=['Geography']))

input_df=pd.concat(objs=[input_df,geo_encoded_df],axis=1)
input_df.drop(columns='Geography',inplace=True)

#Standardization
input_scaled=scaler_obj.transform(input_df)

#prediction
predict_proba=model.predict(input_scaled)[0][0]

st.write(f'Churn Probability:{predict_proba:.2f}')

if predict_proba>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')




