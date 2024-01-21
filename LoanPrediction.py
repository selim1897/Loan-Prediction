import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


st.set_page_config(page_title="Input", page_icon="📈", layout="centered", initial_sidebar_state="auto", menu_items=None)

df = pd.read_csv("Data.csv")

df.drop('Loan_ID',axis=1,inplace=True)

null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']


for col in null_cols:
    df[col] = df[col].fillna(
    df[col].dropna().mode().values[0] )   

    
df.isnull().sum().sort_values(ascending=False)

to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}

df = df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

Dependents = pd.to_numeric(df.Dependents)

df.drop(['Dependents'], axis = 1, inplace = True)

df = pd.concat([df, Dependents], axis = 1)

XGB = XGBClassifier()
y = df['Loan_Status']
X = df.drop('Loan_Status', axis = 1)
XGB.fit(X, y)
            
def prediction():
    data = {'Gender':gender, 'Married':married, 'Dependents':dependents, 'Education':education, 'Self_Employed':self_employed, 'ApplicantIncome':applicant_income,
    'CoapplicantIncome':coapplicant_income, 'LoanAmount':loan_amount, 'Loan_Amount_Term':loan_amount_term, 'Credit_History':credit_history,
    'Property_Area':property_area}

    for key in data.keys():
        if data[key] is None:
            st.session_state['info'] = key + ' is not Valid'
            return
        elif data[key] in to_numeric:
            data[key] = to_numeric[data[key]]

    if 'info' in st.session_state:
        del st.session_state['info']
    
    x_pred = pd.DataFrame(data, index=[0])
    dep = x_pred.Dependents
    x_pred.drop(['Dependents'], axis = 1, inplace = True)
    x_pred = pd.concat([x_pred, dep], axis = 1)
    y_predict = XGB.predict(x_pred)
    if y_predict[0] == 0:
        st.error('We regret to inform you that your loan application has not been approved at this time. ')
    else:
        st.success('Great news! Your loan application has been approved. Our team will contact you soon to discuss the next steps. ')



gender = st.selectbox(
    'Gender',
    ('Male', 'Female'), index=None, placeholder="Choose an option")

married = st.selectbox(
    'Married',
    ('Yes', 'No'), index=None, placeholder="Choose an option")

dependents = st.selectbox(
    'Dependents',
    (0, 1, 2, '3+'), index=None, placeholder="Choose an option")

education = st.selectbox(
    'Education',
    ('Graduate', 'Not Graduate'), index=None, placeholder="Choose an option")

self_employed = st.selectbox(
    'Self Employed',
    ('Yes', 'No'), index=None, placeholder="Choose an option")

applicant_income = st.number_input('Applicant Income')

coapplicant_income = st.number_input('Coapplicant Income')

loan_amount = st.number_input('Loan Amount')

loan_amount_term = st.number_input('Loan Amount Term')

credit_history = st.selectbox(
    'Credit History',
    (0, 1), index=None, placeholder="Choose an option")

property_area = st.selectbox(
    'Property Area',
    ('Urban', 'Semiurban', 'Rural'), index=None, placeholder="Choose an option")

if st.button("Predict", type="primary"):
    prediction()

if 'info' in st.session_state:
    st.info(st.session_state['error'], icon="ℹ️")


