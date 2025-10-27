"""
GDP vs Happiness Regression Module

This module implements linear regression on the GDP vs Happiness dataset
using both Gradient Descent (GD) and Ordinary Least Squares (OLS).

Author: Swesan Pathmanathan
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class GDPRegression:
    """
    Linear Regression model for the GDP vs Happiness dataset.

    Attributes
    ----------
    data_path : str
        Path to the CSV dataset file.
    data_2018 : pd.DataFrame
        Cleaned DataFrame filtered for the year 2018.
    """

    def __init__(self, csv_path: str = "datasets/gdp-vs-happiness.csv") -> None:
        """Initialize GDPRegression with dataset path."""
        self.data_path = csv_path
        self.data_2018 = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess GDP vs Happiness dataset (2018 only)."""
        data = pd.read_csv(self.data_path)
        df = (
            data[data["Year"] == 2018]
            .drop(columns=["World regions according to OWID", "Code"])
            .dropna(subset=["Cantril ladder score", "GDP per capita, PPP (constant 2021 international $)"])
        )
        return df

    def _normalize(self, x: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Normalize an array and return normalized values, mean, and std."""
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / std, mean, std

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract and normalize GDP and Happiness columns."""
        mask = self.data_2018["Cantril ladder score"] > 4.5
        gdp = self.data_2018.loc[mask, "GDP per capita, PPP (constant 2021 international $)"].to_numpy()
        happiness = self.data_2018.loc[mask, "Cantril ladder score"].to_numpy()
        x_norm, _, _ = self._normalize(gdp)
        y_norm, _, _ = self._normalize(happiness)
        X = np.column_stack((np.ones(len(x_norm)), x_norm))
        Y = y_norm.reshape(-1, 1)
        return X, Y

    def fit_gradient_descent(self, X, Y, epochs: int, eta: float) -> np.ndarray:
        """Fit model parameters using gradient descent."""
        beta = np.random.randn(2, 1)
        for _ in range(epochs):
            y_hat = X @ beta
            error = y_hat - Y
            gradient = (2 / len(Y)) * (X.T @ error)
            beta -= eta * gradient
        return beta

    def fit_ols(self, X, Y) -> np.ndarray:
        """Fit model parameters using the OLS closed-form solution."""
        return np.linalg.inv(X.T @ X) @ (X.T @ Y)

    def run(self) -> tuple:
        """Run full gradient descent experiment with grid search."""
        X, Y = self._prepare_data()
        
        # Grid search parameters
        learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        epoch_counts = [200, 500, 1000, 2000, 5000]
        
        results = []
        
        # Grid search over learning rates and epochs
        for eta in learning_rates:
            for epochs in epoch_counts:
                beta = self.fit_gradient_descent(X, Y, epochs, eta)
                
                # Calculate MSE
                y_pred = X @ beta
                mse = float(np.mean((Y - y_pred) ** 2))
                
                results.append((mse, eta, epochs, beta))
        
        # Sort by MSE (ascending)
        results.sort(key=lambda x: x[0])
        
        # Fit OLS for comparison
        beta_ols = self.fit_ols(X, Y)
        
        # Get best GD result
        if results:  # Check if results is not empty
            best_mse, best_eta, best_epochs, best_beta = results[0]
            
            print(f"Best GD Result:")
            print(f"  Learning Rate: {best_eta}")
            print(f"  Epochs: {best_epochs}")
            print(f"  MSE: {best_mse:.6f}")
            
            # Generate and save plot
            fig = self._plot_results(X, Y, results, beta_ols, best_beta)
            
            # Return results for Streamlit display
            return {
                'best_mse': best_mse,
                'best_eta': best_eta,
                'best_epochs': best_epochs,
                'figure': fig,
                'X': X,
                'Y': Y
            }
        else:
            print("No gradient descent results generated!")
            return None

    def _plot_results(self, X, Y, results, beta_ols, best_beta):
        """Plot top 5 GD results, OLS, and best GD fit with timestamp filename."""
        if not results:  # Safety check
            print("No results to plot!")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of data
        ax.scatter(X[:, 1], Y, alpha=0.6, color='blue', label='Data Points')
        
        # Plot top 5 GD lines
        x_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        X_range = np.column_stack([np.ones(100), x_range])
        
        for i, (mse, eta, epochs, beta) in enumerate(results[:5]):
            if beta.size > 0:  # Check if beta is not empty
                y_pred = X_range @ beta
                ax.plot(x_range, y_pred, '--', alpha=0.7, 
                       label=f'GD {i+1} (η={eta}, MSE={mse:.3f})')
        
        # Plot OLS line
        if beta_ols.size > 0:  # Check if beta_ols is not empty
            y_ols = X_range @ beta_ols
            ax.plot(x_range, y_ols, 'r-', linewidth=2, label='OLS')
        
        # Highlight best GD result
        if best_beta.size > 0:  # Check if best_beta is not empty
            y_best = X_range @ best_beta
            ax.plot(x_range, y_best, 'g-', linewidth=3, label='Best GD')
        
        ax.set_xlabel('GDP per capita (normalized)')
        ax.set_ylabel('Happiness Score (normalized)')
        ax.set_title('GDP vs Happiness: Gradient Descent vs OLS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Create timestamp filename and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gdp_regression_results_{timestamp}.png"
        
        # Ensure plots directory exists
        os.makedirs("plots", exist_ok=True)
        filepath = os.path.join("plots", filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        print(f"Plot saved to: {filepath}")
        
        # Return figure for Streamlit display
        return fig
