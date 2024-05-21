import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Input", page_icon="📈", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.image('Tek-up_Logo.png')

to_numeric = {' Yes': 1, ' No': 2, ' Graduate': 1, ' Not Graduate': 2, ' Approved': 1, ' Rejected': 0}

num_cols_normalization = [' income_annual', ' loan_amount', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']

if 'xgb' not in st.session_state:
    xgb = joblib.load('xgb_model.pkl')

    st.session_state['xgb'] = xgb
else:
    xgb = st.session_state['xgb']

if 'scaler' not in st.session_state:
    scaler = joblib.load('scaler_minmax.pkl')

    st.session_state['scaler'] = scaler
else:
    scaler = st.session_state['scaler']
            
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

    variables_to_scale = [' income_annual', ' loan_amount', ' loan_term',' cibil_score', ' residential_assets_value', ' commercial_assets_value', 
                      ' luxury_assets_value', ' bank_asset_value']

    x_pred[variables_to_scale] = scaler.transform(x_pred[variables_to_scale])


    y_predict = xgb.predict(x_pred)
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
