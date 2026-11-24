import streamlit as st
import joblib

# Saved model aur vectorizer load karein
model = joblib.load('Churn_predic_model.pkl')

# Website UI
st.set_page_config(page_title="Intelli_serve_india", page_icon="🛍️")
st.title("🛍️ Churn Customer Prediction")
st.write("Enter a customerplan below to predict whether the customer churn or not.")

user_review = st.text_area("Customer Contract Type:", placeholder="Example: Monthly, Annualy, Quaterly")
user_review = st.text_area("Customer Days Since Last Activity:")
user_review = st.text_area("How Many Complaint Ticket Raised:")
user_review = st.text_area("Customer Satisfaction Score:")

if st.button("Predict churn"):
        if prediction == 1:
            st.success("✅ **Churned** (Customer is unsatisfied)")
        else:
            st.error("❌ **Not Churned** (Customer is satisfied)")
