"""
Streamlit Dashboard for GDP & Abalone Regression Models

Launch interactive dashboards to visualize and compare regression fits.

Run: streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
from models.gdp_regression import GDPRegression
from models.abalone_regression import AbaloneRegression


st.set_page_config(page_title="GDP & Abalone Regression", layout="wide")
st.title("GDP and Abalone Regression Explorer")

tab1, tab2 = st.tabs(["GDP vs Happiness", "Abalone Regression"])

# --- GDP vs Happiness Tab ---
with tab1:
    st.header("Linear Regression — GDP vs Happiness")
    eta = st.select_slider("Learning Rate (η)", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)
    epochs = st.slider("Epochs", 200, 5000, 1000, step=200)

    if st.button("Run GDP Regression"):
        model = GDPRegression()
        X, Y = model._prepare_data()
        beta = model.fit_gradient_descent(X, Y, epochs, eta)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X[:, 1], Y, label="Data", alpha=0.7)
        ax.plot(sorted(X[:, 1]), sorted((X @ beta).ravel()), color="g", label="Best GD Fit")
        ax.legend()
        st.pyplot(fig)

# --- Abalone Tab ---
with tab2:
    st.header("Polynomial Regression — Abalone Dataset")
    max_degree = st.slider("Max Polynomial Degree", 1, 6, 3)

    if st.button("Run Abalone Regression"):
        model = AbaloneRegression()
        model.train(max_degree=max_degree)
        st.success("Model training completed. Check the plots directory for generated figures.")
