"""
Abalone Polynomial Regression Module

Implements polynomial regression with degree selection, feature scaling,
and evaluation on the Abalone dataset.

Author: Swesan Pathmanathan
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AbaloneRegression:
    """
    Polynomial regression on the Abalone dataset.

    Attributes
    ----------
    csv_path : str
        Path to the dataset CSV.
    data : pd.DataFrame
        Full dataset including feature and target columns.
    """

    def __init__(self, csv_path: str = "datasets/training_data.csv") -> None:
        """Initialize AbaloneRegression with dataset path."""
        self.csv_path = csv_path
        self.data = pd.read_csv(self.csv_path, index_col=0)
        self.target_col = "Rings"

    @staticmethod
    def _standardize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return standardized data and its mean/std."""
        mu, sigma = np.mean(x), np.std(x)
        if sigma == 0:
            sigma = 1.0
        return (x - mu) / sigma, mu, sigma

    def _split(self, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/test subsets."""
        n = len(self.data)
        test_size = int((1 - frac) * n)
        perm = np.random.default_rng(42).permutation(n)
        test_idx = perm[:test_size]
        train_idx = perm[test_size:]
        return self.data.iloc[train_idx], self.data.iloc[test_idx]

    def _design_matrix(self, df, features, degrees, stats):
        """Construct standardized polynomial design matrix."""
        n = len(df)
        X = [np.ones(n)]
        names = ["Intercept"]
        for feat in features:
            x = df[feat].to_numpy()
            mu, sigma = stats[feat]
            x_std = (x - mu) / sigma
            for p in range(1, degrees.get(feat, 1) + 1):
                X.append(x_std ** p)
                names.append(f"{feat}^{p}" if p > 1 else feat)
        return np.column_stack(X), names

    def train(self, max_degree: int = 4) -> None:
        """Fit polynomial regression per feature and plot results."""
        train_df, test_df = self._split()
        features = [c for c in train_df.columns if c != self.target_col]
        Y_train = train_df[self.target_col].to_numpy() + 1.5
        Y_test = test_df[self.target_col].to_numpy() + 1.5

        # Compute feature stats
        stats = {f: (train_df[f].mean(), train_df[f].std() or 1) for f in features}

        # Select degree per feature via validation
        degrees = {f: 1 for f in features}
        best_mse = float("inf")
        for f in features:
            for d in range(1, max_degree + 1):
                local_deg = {f: d}
                X_train, _ = self._design_matrix(train_df, features, local_deg, stats)
                beta = np.linalg.pinv(X_train) @ Y_train
                mse = float(np.mean((Y_train - X_train @ beta) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    degrees[f] = d

        # Final fit
        X_train, names = self._design_matrix(train_df, features, degrees, stats)
        X_test, _ = self._design_matrix(test_df, features, degrees, stats)
        beta = np.linalg.pinv(X_train) @ Y_train

        train_mse = float(np.mean((Y_train - X_train @ beta) ** 2))
        test_mse = float(np.mean((Y_test - X_test @ beta) ** 2))
        print(f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

        self._plot_results(features, train_df, test_df, degrees, stats, beta)

    def _plot_results(self, features, train_df, test_df, degrees, stats, beta):
        """Generate plots for each feature."""
        os.makedirs("plots", exist_ok=True)
        cols = 3
        rows = int(np.ceil(len(features) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            x_train = train_df[feat].to_numpy()
            y_train = train_df[self.target_col].to_numpy() + 1.5
            mu, sigma = stats[feat]
            deg = degrees[feat]
            x_sorted = np.sort(x_train)
            x_std = (x_sorted - mu) / sigma
            X_poly = np.column_stack([np.ones_like(x_std)] + [x_std ** p for p in range(1, deg + 1)])
            y_pred = X_poly @ beta[:deg + 1]

            ax.scatter(x_train, y_train, s=15, alpha=0.6, label="Train")
            ax.plot(x_sorted, y_pred, color="r", label=f"Degree {deg}")
            ax.set_xlabel(feat)
            ax.set_ylabel("Rings")
            ax.legend()

        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig("plots/abalone_regression_results.png", dpi=300)
        plt.close()
