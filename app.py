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
        with st.spinner("Running regression analysis..."):
            model = GDPRegression()
            results = model.run()
            
            if results:
                # Display results
                st.success("GDP regression completed!")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Best Gradient Descent Result:")
                    st.write(f"**Learning Rate:** {results['best_eta']}")
                    st.write(f"**Epochs:** {results['best_epochs']}")
                    st.write(f"**MSE:** {results['best_mse']:.6f}")
                
                with col2:
                    st.subheader("Regression Analysis")
                    if results['figure']:
                        st.pyplot(results['figure'])
                        plt.close()  # Clean up
            else:
                st.error("Failed to generate regression results!")

# --- Abalone Tab ---
with tab2:
    st.header("Polynomial Regression — Abalone Dataset")
    max_degree = st.slider("Max Polynomial Degree", 1, 6, 3)

    if st.button("Run Abalone Regression"):
        with st.spinner("Training polynomial regression..."):
            model = AbaloneRegression()
            results = model.train(max_degree=max_degree)
            
            if results:
                # Display results
                st.success("Abalone regression completed!")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Model Performance:")
                    st.write(f"**Train MSE:** {results['train_mse']:.4f}")
                    st.write(f"**Test MSE:** {results['test_mse']:.4f}")
                    
                    st.subheader("Selected Polynomial Degrees:")
                    for feature, degree in results['degrees'].items():
                        st.write(f"**{feature}:** {degree}")
                
                with col2:
                    st.subheader("Per-Feature Polynomial Fits")
                    if results['figure']:
                        st.pyplot(results['figure'])
                        plt.close()  # Clean up
            else:
                st.error("Failed to generate regression results!")
