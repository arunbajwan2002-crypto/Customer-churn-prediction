
import streamlit as st
import joblib
import pandas as pd


model = joblib.load('Churn_prediction_model.pkl')
st.set_page_config(page_title = 'Intelli serve india', page_icon = '00')
st.title('Customer Churn Prediction Model')
st.write('Enter Customer details to find out the churn probablitity either customer churn or not')

col1, col2 = st.columns(2)

with col1:
    contract_type = st.selectbox('Contract type', ['Annualy', 'Quaterly', 'Monthly'])
    
    Days_Since_Last_Activity = st.number_input('Last  activity', min_value = 0, max_value = 30, value = 1)
   

with col2:
    Complaint_Ticket_raised = st.slider('Ticket raised', min_value = 0, max_value = 8, value = 1)
    Satisfaction_Score = st.slider('Satisfaction score', min_value = 0, max_value = 5, value = 1)
    

contract_monthly = 1 if contract_type == 'Monthly' else 0
contract_quaterly = 1 if contract_type == 'Quaterly' else 0

if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
            'Tenure_months':12,
            'Days_Since_Last_Activity':Days_Since_Last_Activity,
            'Monthly_Spend': 3500,
            'Complaint_Ticket_Raised': Complaint_Ticket_raised,
            'Satisfaction_Score': Satisfaction_Score,
            'Usage_Drop': 0.0,
            'Contract_Type_Monthly': contract_monthly,
            'Contract_Type_Quaterly': contract_quaterly,
           
        }])


    churn_prob = model.predict(input_df)[0]
    customer_threshold = 0.45

    prediction = 1 if churn_prob > customer_threshold else 0

    st.markdown('-----')
    if prediction == 1:
        st.error(f'High Risk--- churned probability: {churn_prob*100: .2f}%')
    else:
        st.success(f'Low Risk--- churned probabilty: {churn_prob*100: .2f}%')




