# GDP and Abalone Insights: Linear & Polynomial Regression Models

This project demonstrates the application of **machine learning regression techniques** on two datasets:

1. **GDP vs Happiness (2018)** – Simple Linear Regression using **Gradient Descent** and **Ordinary Least Squares (OLS)**  
2. **Abalone Dataset** – Polynomial Regression with **automatic per-feature degree selection**, **feature scaling**, and **model evaluation**

An interactive dashboard built with **Streamlit** allows users to run regression experiments and view results alongside automatically saved timestamped plots.

---

## 🧠 Project Overview

This project was originally developed as part of an applied machine learning course and later extended into a standalone repository.  
It explores how simple and polynomial regression techniques can be applied to real-world datasets to uncover patterns and make predictions.

### Key Concepts Demonstrated

- Linear Regression using Gradient Descent (5×5 grid search) and OLS  
- Polynomial Regression with automatic per-feature degree selection on training data
- Feature scaling using z-score standardization  
- Train/test data splitting (80/20)  
- Model evaluation using Mean Squared Error (MSE)  
- Data visualization on original vs standardized scales
- Timestamped plot generation and file management
- Interactive experiment running via Streamlit  

---

## 📂 Repository Structure

```
gdp-abalone-regression-ml/
│
├── app.py                      # Streamlit dashboard entry point
│
├── models/
│   ├── gdp_regression.py       # Linear regression model (GDP vs Happiness)
│   └── abalone_regression.py   # Polynomial regression model (Abalone dataset)
│
├── datasets/
│   ├── gdp-vs-happiness.csv    # GDP and happiness scores by country (2018)
│   └── training_data.csv       # Abalone physical measurements and age data
│
├── plots/                      # Generated timestamped output images
│   ├── gdp_regression_results_YYYYMMDD_HHMMSS.png
│   └── abalone_regression_results_YYYYMMDD_HHMMSS.png
│
├── environment.yml             # Conda environment file
├── README.md                   # Project documentation
├── LICENSE
└── .gitignore
```

---

## ⚙️ Model Specifications

### GDP vs Happiness — Linear Regression

**Technical Details:**
- **Features:** GDP per capita (PPP, constant 2021$)  
- **Target:** Cantril Ladder Happiness Score (2018)
- **Preprocessing:** Z-score normalization for both features and targets
- **Optimization:** 
  - **Gradient Descent:** 5×5 grid search over learning rates [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] and epochs [200, 500, 1000, 2000, 5000]
  - **OLS:** Closed-form solution using pseudoinverse
- **Evaluation:** Mean Squared Error (MSE) on normalized data
- **Output:** Top 5 GD results + OLS comparison + best GD highlighted

### Abalone Dataset — Polynomial Regression

**Technical Details:**
- **Dataset Size:** 2,577 samples × 8 features
- **Features:** `['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']`
- **Target:** `Rings + 1.5` (age in years)
- **Preprocessing:** Z-score standardization per feature
- **Train/Test Split:** 80/20 with fixed random seed (42)
- **Hyperparameters:**
  - `max_degree`: 1-6 (default: 4)
  - `train_split`: 0.8 (80% training data)
  - `random_seed`: 42 (reproducible splits)
- **Degree Selection:** Greedy per-feature selection (1 to `max_degree`) based on training MSE
- **Method:** Individual polynomial fits per feature (not multivariate polynomial)
- **Evaluation:** Train and Test MSE
- **Visualization:** Per-feature polynomial curves on original data scale

---

### 🧩 Installing Conda

If you don't already have Conda installed, you can install it using **[Miniconda](https://www.anaconda.com/download/success)**

#### Miniconda (lightweight, recommended)

**Windows / macOS / Linux:**

1. Go to [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)
2. Download the installer for your OS and Python 3.x (64-bit)
3. Run the installer:
   - On **Windows**, open the `.exe` and follow on-screen instructions  
   - On **macOS/Linux**, run:

     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     bash Miniconda3-latest-Linux-x86_64.sh
     ```

4. After installation, restart your terminal and verify:

   ```bash
   conda --version
   ```

Once Conda is installed, continue with the setup:

```bash
conda env create -f environment.yml
conda activate gdp-abalone-ml
```

---

## 🚀 Running the Scripts

### 1. Linear Regression (GDP vs Happiness)

Performs linear regression using both **Gradient Descent** and **OLS** methods.  
Runs 5×5 grid search over learning rates and epochs, then visualizes top 5 GD results with OLS comparison.

```bash
python3 models/gdp_regression.py
```

**Output:**

- Prints best GD parameters and MSE
- Saves timestamped plot: `plots/gdp_regression_results_YYYYMMDD_HHMMSS.png`
- Plot includes: Top 5 GD lines, OLS line (red), Best GD (green), data scatter

---

### 2. Polynomial Regression (Abalone Dataset)

Performs polynomial regression with automatic per-feature degree selection.  
Includes train/test split, z-score standardization, degree tuning on training data, and per-feature visualization.

```bash
python3 models/abalone_regression.py
```

**Output:**

- Prints Train/Test MSE and selected degrees per feature
- Saves timestamped plot: `plots/abalone_regression_results_YYYYMMDD_HHMMSS.png`
- Plot shows: 7 subplots with individual feature polynomials on original scale

---

## 📊 Example Outputs

| Dataset | Method | Output File | Contains |
|----------|------------|-------------|----------|
| GDP vs Happiness | 5×5 Grid Search + OLS | `gdp_regression_results_YYYYMMDD_HHMMSS.png` | Scatter plot + top 5 GD lines + OLS + best GD |
| Abalone | Per-feature Polynomials | `abalone_regression_results_YYYYMMDD_HHMMSS.png` | 7 subplots showing individual feature polynomial fits |

**Sample Console Output:**
```
# GDP Regression
Best GD Result:
  Learning Rate: 0.001
  Epochs: 5000
  MSE: 0.473046

# Abalone Regression
Available features: ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
Train MSE: 4.8234
Test MSE: 5.1267
Selected degrees: {'Length': 3, 'Diameter': 2, 'Height': 4, 'Whole_weight': 2, 'Shucked_weight': 3, 'Viscera_weight': 2, 'Shell_weight': 3}
```

---

## 🧩 Streamlit Dashboard

A basic interactive dashboard for running regression experiments with parameter adjustment.

To launch:

```bash
streamlit run app.py
```

**Features:**

- **GDP Tab:** 
  - Adjustable learning rate slider (1e-5 to 1e-3)
  - Epochs slider (200 to 5000)
  - Button to run full grid search
  - Displays best result parameters and MSE
  - Shows regression plot inline

- **Abalone Tab:** 
  - Max polynomial degree slider (1 to 6)
  - Button to run training with degree selection
  - Displays train/test MSE and selected degrees
  - Shows per-feature polynomial plots inline

**Output Behavior:**
- Results displayed in Streamlit interface
- **AND** timestamped plots automatically saved to `plots/` directory
- Console output shows detailed parameters and file paths

**Limitations:**
- No real-time parameter updates (requires button clicks)
- No cross-validation or advanced model selection
- Basic UI with limited customization options

---

## 🧮 Datasets

| Dataset | Description | Size | Features | Target | Source |
|----------|--------------|------|----------|---------|---------|
| `gdp-vs-happiness.csv` | Global GDP per capita vs Happiness scores (2018) | Varies by year | GDP per capita (PPP, 2021$) | Cantril Ladder Score | World Happiness Report 2018 |
| `training_data.csv` | Abalone physical measurements | 2,577 × 8 | Length, Diameter, Height, 4× Weight measurements | Rings (age proxy) | UCI Machine Learning Repository |

**Abalone Dataset Details:**
- **Physical measurements** in continuous values (inches/grams)
- **Target transformation:** `Age = Rings + 1.5 years`
- **Missing values:** None (pre-cleaned dataset)
- **Feature ranges:** Length (0.075-0.815), Weights (0.002-2.826), etc.

---

## 📈 Technical Implementation

### GDP Model Architecture
```
Input: GDP per capita (1D)
↓ Z-score normalization
↓ Design matrix: [1, x] 
↓ Grid search: η ∈ {1e-5,...,1e-3}, epochs ∈ {200,...,5000}
↓ Gradient descent: θ = θ - η∇J(θ)
↓ Compare with OLS: θ = (X^T X)^(-1) X^T y
Output: Best parameters + MSE comparison
```

### Abalone Model Architecture
```
Input: 7 continuous features
↓ 80/20 train/test split (seed=42)
↓ Per-feature z-score standardization
↓ For each feature f, for each degree d ∈ {1,...,max_degree}:
    ↓ Train polynomial: β = (X^T X)^(-1) X^T y
    ↓ Select best degree by training MSE
↓ Final model: concatenate all feature polynomials
↓ Evaluate on test set
Output: Per-feature degrees + train/test MSE
```

---

## 🧰 Tools and Technologies

- **Languages:** Python 3.10  
- **Core Libraries:** NumPy (linear algebra), Pandas (data manipulation), Matplotlib (visualization)
- **Framework:** Streamlit (interactive dashboard)
- **Environment:** Conda (reproducible dependencies)  
- **Development:** GitHub Copilot and Claude Sonnet (code assistance and documentation)

---

## ⚡ Estimated Development Carbon Footprint

Approx. **3.5 kg CO₂**, based on 10 hours of AI-assisted coding, documentation, and testing.

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**S. Pathmanathan**  
5th Year Software Engineering Student @ McMaster University  
Prev @ AMD (Datacenter GPU Validation)  
*Turning ideas into code that actually makes life easier.*
