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

    def run(self) -> None:
        """Execute full regression experiment and save results."""
        X, Y = self._prepare_data()
        etas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        epochs = [200, 500, 1000, 2000, 5000]
        results = []

        for eta in etas:
            for epoch in epochs:
                beta = self.fit_gradient_descent(X, Y, epoch, eta)
                mse = float(np.mean((X @ beta - Y) ** 2))
                results.append((mse, beta, eta, epoch))

        results.sort(key=lambda x: x[0])
        best_mse, best_beta, best_eta, best_epoch = results[0]
        beta_ols = self.fit_ols(X, Y)

        self._plot_results(X, Y, results, beta_ols, best_beta)
        print(f"Best GD -> η={best_eta}, epochs={best_epoch}, MSE={best_mse:.4f}")

    def _plot_results(self, X, Y, results, beta_ols, best_beta):
        """Plot GD lines, OLS comparison, and save figures."""
        os.makedirs("plots", exist_ok=True)
        X_ = X[:, 1]
        idx = np.argsort(X_)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_, Y, label="Data", alpha=0.8)

        for mse, beta, eta, epoch in results[:5]:
            ax.plot(X_[idx], (X @ beta).ravel()[idx], label=f"η={eta}, epoch={epoch}")

        ax.plot(X_[idx], (X @ beta_ols).ravel()[idx], color="r", lw=2, label="OLS")
        ax.plot(X_[idx], (X @ best_beta).ravel()[idx], color="g", lw=2, label="Best GD")
        ax.legend()
        ax.set_title("GDP vs Happiness Regression")
        ax.set_xlabel("GDP per capita")
        ax.set_ylabel("Happiness Score")
        plt.tight_layout()
        plt.savefig("plots/gdp_regression_results.png", dpi=300)
        plt.close()
