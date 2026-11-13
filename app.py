import streamlit as st
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('lr.sav1')

model = load_model()

# App title and description
st.title("📈 Stock Price Prediction App")
st.write("""
This app predicts the **Stock Price** based on financial metrics using a trained Linear Regression model.
Provide the inputs in the sidebar to get predictions.
""")

# Sidebar input fields
st.sidebar.header("Input Stock Metrics")

eps = st.sidebar.number_input("Earnings Per Share (EPS)", min_value=-50.0, max_value=200.0, value=25.0, step=0.1)
book_value = st.sidebar.number_input("Book Value", min_value=0.0, max_value=1000.0, value=100.0, step=0.1)
st_roe = st.sidebar.slider("Return on Equity (ROE %)", min_value=-20.0, max_value=100.0, value=15.0, step=0.5)
debt_equity = st.sidebar.slider("Debt to Equity Ratio", min_value=0.0, max_value=10.0, value=0.5, step=0.05)
sales_growth = st.sidebar.slider("Sales Growth (%)", min_value=-50.0, max_value=100.0, value=10.0, step=0.5)

# Predict Button
if st.button("Predict Stock Price"):
    # Format inputs as a numpy array with matching features shape (1, 5)
    features = np.array([[eps, book_value, st_roe, debt_equity, sales_growth]])
    
    # Generate prediction
    prediction = model.predict(features)[0]
    
    # Display predictions
    st.success(f"### 🎯 Predicted Stock Price: **₹{prediction:.2f}**")
    
    # Display details
    st.write("**Input Summary:**")
    st.json({
        "EPS": eps,
        "Book Value": book_value,
        "ROE (%)": st_roe,
        "Debt/Equity": debt_equity,
        "Sales Growth (%)": sales_growth
    })
