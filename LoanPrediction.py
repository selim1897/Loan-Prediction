import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Input", page_icon="📈", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.image('Tek-up_Logo.png')

to_numeric = {' Yes': 1, ' No': 2, ' Graduate': 1, ' Not Graduate': 2, ' Approved': 1, ' Rejected': 0}

num_cols_normalization = [' income_annual', ' loan_amount', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']

@st.cache_resource
def load_and_preprocess_data():
    xgb = {}
    
    df = pd.read_csv("Data.csv")
    df.drop('loan_id',axis=1,inplace=True)

    df = df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

    for col in num_cols_normalization:
        scaler_minmax = MinMaxScaler()
        scaler_minmax.fit(df[col].values.reshape(-1, 1))
        xgb[col] = scaler_minmax
        df[col] = scaler_minmax.transform(df[col].values.reshape(-1, 1))

    XGB = XGBClassifier()
    y = df[' loan_status']
    X = df.drop(' loan_status', axis=1)
    XGB.fit(X, y)

    xgb['XGB'] = XGB

    return xgb

if 'xgb' not in st.session_state:
    xgb = load_and_preprocess_data()

    st.session_state['xgb'] = xgb
else:
    xgb = st.session_state['xgb']
            
def prediction():
    data = {' no_of_dependents':dependents, ' education':education, ' self_employed':self_employed, ' income_annual':income_annual, ' loan_amount':loan_amount, 
    ' loan_term':loan_amount_term, ' cibil_score':cibil_score, ' residential_assets_value':residential_assets_value, ' commercial_assets_value':commercial_assets_value, 
    ' luxury_assets_value':luxury_assets_value, ' bank_asset_value':bank_asset_value}

    for key in data.keys():
        if data[key] is None:
            st.session_state['info'] = key + ' is not Valid'
            return
        elif data[key] in to_numeric:
            data[key] = to_numeric[data[key]]

    if 'info' in st.session_state:
        del st.session_state['info']

    
    x_pred = pd.DataFrame(data, index=[0])

    for col in num_cols_normalization:
        x_pred[col] = xgb[col].transform(x_pred[col].values.reshape(-1, 1))


    y_predict = xgb['XGB'].predict(x_pred)
    if y_predict[0] == 0:
        st.error('We regret to inform you that your loan application has not been approved at this time. ')
    else:
        st.success('Great news! Your loan application has been approved. Our team will contact you soon to discuss the next steps. ')



dependents = st.selectbox(
    ' no_of_dependents',
    (0, 1, 2, 3, 4, 5), index=None, placeholder="Choose an option")

education = st.selectbox(
    'Education',
    (' Graduate', ' Not Graduate'), index=None, placeholder="Choose an option")

self_employed = st.selectbox(
    'Self Employed',
    (' Yes', ' No'), index=None, placeholder="Choose an option")

income_annual = st.number_input('Income_annual')

loan_amount = st.number_input('Loan Amount')

loan_amount_term = st.number_input('Loan Amount Term', step=1)

cibil_score = st.number_input('cibil_score', min_value=0, max_value=1000, step=1)

residential_assets_value = st.number_input('residential_assets_value')

commercial_assets_value	= st.number_input('commercial_assets_value')

luxury_assets_value	= st.number_input('luxury_assets_value')

bank_asset_value = st.number_input('bank_asset_value')


if st.button("Predict", type="primary"):
    prediction()

if 'info' in st.session_state:
    st.info(st.session_state['info'], icon="ℹ️")
