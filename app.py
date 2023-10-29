import streamlit as st
import pandas as pd
import numpy as np

from pycaret.regression import *

model=load_model('final_model_14Oct2023')


def user_input_processed(df):
  return df

def get_user_input():
  """
  This function is used to get user input
  """
  category = st.selectbox("Select Category", ('Category 40', 'Category 20'))
  subcategory = st.selectbox("Select SubCategory", ('SubCategory 215', 'SubCategory 125'))
  location    = st.selectbox("Location", ('Location 165', 'Location 93'))
  opened_by   = st.selectbox("Opened By", ('Opened by 397', 'Opened by 180'))
  resolved_by = st.selectbox("Resolved by", ('Resolved by 81', 'Resolved by 62'))

  features = { 'category' : category,
  	          'subcategory' : subcategory,
              'location' : location,
              'opened_by' : opened_by,
              'resolved_by' : resolved_by
              }
  data = pd.DataFrame(features, index=[0])
  return data


def visualize_output(prediction_proba):
  """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
  """
  st.write(prediction_proba)
  


# Title of the web app
st.title("Incident Management SLA predictor")

st.write("This web app predicts the probablity of meeting SLA given the following input parameters")

user_input = get_user_input()
user_input_processed = user_input_processed(user_input)

st.subheader('User Input parameters')
st.write(user_input_processed)

prediction = model.predict(user_input_processed)
predict_proba = model.predict_proba(user_input_processed)

visualize_output(predict_proba)
