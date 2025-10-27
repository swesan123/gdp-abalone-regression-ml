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
import os
from datetime import datetime


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
        """
        Create design matrix with polynomial terms for given degrees.
        """
        X = [np.ones(len(df))]  # Bias term
        
        for feat in features:
            degree = degrees.get(feat, 1)
            # Fix: stats[feat] is a tuple (standardized_array, mean, std)
            _, mu, sigma = stats[feat]  # Unpack all 3 values
            
            # Get standardized feature values
            x_std = df[feat].values
            
            # Add polynomial terms
            for d in range(1, degree + 1):
                X.append(x_std ** d)
        
        return np.column_stack(X)

    def train(self, max_degree: int = 4) -> dict:
        """Train polynomial regression with degree selection."""
        # Get actual feature columns (exclude target column)
        all_columns = self.data.columns.tolist()
        features = [col for col in all_columns if col != self.target_col]
        
        print(f"Available features: {features}")
        print(f"Target column: {self.target_col}")
        
        train_df, test_df = self._split()
        
        # Keep original data for plotting
        train_df_original = train_df.copy()
        test_df_original = test_df.copy()
        
        # Fix pandas warning by creating copies
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Standardize features
        stats = {}
        for feat in features:
            train_df[feat], mean_val, std_val = self._standardize(train_df[feat].values)
            test_df[feat] = (test_df[feat].values - mean_val) / std_val
            stats[feat] = (None, mean_val, std_val)  # Store as tuple (standardized_array, mean, std)
    
        # Degree selection per feature (on training data)
        degrees = {}
        for feat in features:
            best_mse = float('inf')
            for d in range(1, max_degree + 1):
                local_deg = {feat: d}
                X_train = self._design_matrix(train_df, features, local_deg, stats)
                Y_train = train_df[self.target_col].values
                
                beta = np.linalg.pinv(X_train) @ Y_train
                mse = float(np.mean((Y_train - X_train @ beta) ** 2))
                
                if mse < best_mse:
                    best_mse = mse
                    degrees[feat] = d
    
        # Train final model with selected degrees
        X_train = self._design_matrix(train_df, features, degrees, stats)
        X_test = self._design_matrix(test_df, features, degrees, stats)
        Y_train, Y_test = train_df[self.target_col].values, test_df[self.target_col].values
    
        beta = np.linalg.pinv(X_train) @ Y_train
    
        # Evaluate
        train_mse = float(np.mean((Y_train - X_train @ beta) ** 2))
        test_mse = float(np.mean((Y_test - X_test @ beta) ** 2))
    
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Selected degrees: {degrees}")
    
        # Generate plot using original (unstandardized) data
        fig = self._plot_results(features, train_df_original, test_df_original, degrees, stats, beta)
    
        # Return results for Streamlit display
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'degrees': degrees,
            'figure': fig,
            'features': features
        }

    def _plot_results(self, features, train_df_orig, test_df_orig, degrees, stats, beta):
        """Plot individual polynomial fits for each feature with timestamp filename."""
        n_features = len(features)
        cols = 2
        rows = (n_features + cols - 1) // cols
    
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if n_features == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
    
        axes = axes.flatten()
    
        for i, feat in enumerate(features):
            ax = axes[i]
            
            # Plot training data (original scale)
            x_train_orig = train_df_orig[feat].values
            y_train_orig = train_df_orig[self.target_col].values + 1.5
            ax.scatter(x_train_orig, y_train_orig, alpha=0.6, s=30, label='Train Data', color='blue')
            
            # Plot test data (original scale)
            x_test_orig = test_df_orig[feat].values
            y_test_orig = test_df_orig[self.target_col].values + 1.5
            ax.scatter(x_test_orig, y_test_orig, alpha=0.6, s=30, color='orange', label='Test Data')
            
            # Generate polynomial fit line using original scale
            x_min = min(x_train_orig.min(), x_test_orig.min())
            x_max = max(x_train_orig.max(), x_test_orig.max())
            x_range_orig = np.linspace(x_min, x_max, 200)
            
            # Standardize the range for prediction
            mean_val = stats[feat][1]
            std_val = stats[feat][2]
            x_range_std = (x_range_orig - mean_val) / std_val
            
            # Create design matrix for this feature's polynomial
            degree = degrees[feat]
            
            # Build feature matrix for all features, but only vary this one
            # For other features, use mean values (0 in standardized space)
            n_points = len(x_range_orig)
            design_cols = [np.ones(n_points)]  # bias
            
            feature_idx = 0
            for f in features:
                f_degree = degrees[f]
                if f == feat:
                    # Use the varying values for current feature
                    for d in range(1, f_degree + 1):
                        design_cols.append(x_range_std ** d)
                else:
                    # Use mean (0) for other features
                    for d in range(1, f_degree + 1):
                        design_cols.append(np.zeros(n_points))
        
            X_pred = np.column_stack(design_cols)
            
            # Make prediction
            if X_pred.shape[1] <= len(beta):
                y_pred = X_pred @ beta[:X_pred.shape[1]]
            else:
                y_pred = X_pred[:, :len(beta)] @ beta
            
            ax.plot(x_range_orig, y_pred, 'r-', linewidth=2, 
                   label=f'Polynomial (degree {degree})')
            
            ax.set_xlabel(f'{feat}')
            ax.set_ylabel('Age (years)')
            ax.set_title(f'{feat} vs Age (Original Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits
            ax.set_xlim(x_min * 0.95, x_max * 1.05)
    
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
    
        # Create timestamp filename and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"abalone_regression_results_{timestamp}.png"
    
        # Ensure plots directory exists
        os.makedirs("plots", exist_ok=True)
        filepath = os.path.join("plots", filename)
    
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
        print(f"Plot saved to: {filepath}")
    
        # Return figure for Streamlit display
        return fig


# Quick debug script to see your dataset structure
import pandas as pd
data = pd.read_csv("datasets/training_data.csv", index_col=0)
print("Dataset shape:", data.shape)
print("Column names:", data.columns.tolist())
print("First few rows:")
print(data.head())
