# GDP and Abalone Insights: Linear & Polynomial Regression Models

This project demonstrates the application of **machine learning regression techniques** on two datasets:

1. **GDP vs Happiness (2018)** – Simple Linear Regression using **Gradient Descent** and **Ordinary Least Squares (OLS)**  
2. **Abalone Dataset** – Polynomial Regression with **automatic degree selection**, **feature scaling**, and **model evaluation**

An interactive dashboard built with **Streamlit** allows users to visualize model predictions, regression fits, and dataset relationships in real time.

---

## 🧠 Project Overview

This project was originally developed as part of an applied machine learning course and later extended into a standalone repository.  
It explores how simple and polynomial regression techniques can be applied to real-world datasets to uncover patterns and make predictions.

### Key Concepts Demonstrated

- Linear Regression using Gradient Descent and OLS  
- Polynomial Regression with automatic degree selection on training data
- Feature scaling and standardization  
- Train/test data splitting  
- Model evaluation using Mean Squared Error (MSE)  
- Data visualization using Matplotlib  
- Interactive exploration using Streamlit  

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
│   ├── gdp-vs-happiness.csv
│   └── training_data.csv
│
├── plots/                      # Generated output images (created at runtime)
│
├── environment.yml             # Conda environment file
│
├── README.md                   # Project documentation
├── LICENSE
└── .gitignore
```

---

### 🧩 Installing Conda

If you don't already have Conda installed, you can install it using **[Miniconda](https://www.anaconda.com/download/success)**

#### Miniconda (lightweight, recommended)

**Windows / macOS / Linux:**

1. Go to [https://www.anaconda.com/download/success](https://www.anaconda.com/download/successl)
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
Tests a 5×5 grid of learning rates and epochs, then plots the top 5 GD results alongside OLS and best GD fits.

```bash
python3 models/gdp_regression.py
```

**Output:**

- Prints best GD parameters and MSE
- Saves plot in `plots/gdp_regression_results.png` showing:
  - Top 5 GD regression lines
  - OLS comparison line
  - Best GD result highlighted

---

### 2. Polynomial Regression (Abalone Dataset)

Performs polynomial regression on the **Abalone** dataset with automatic degree selection per feature.  
Includes train/test split, standardization, training-based degree tuning, and visualization.

```bash
python3 models/abalone_regression.py
```

**Output:**

- Prints Train/Test MSE results
- Saves visualization in `plots/abalone_regression_results.png` showing per-feature polynomial fits

---

## 📊 Example Outputs

| Dataset | Technique | Output File | Description |
|----------|------------|-------------|--------------|
| GDP vs Happiness | Gradient Descent + OLS | `plots/gdp_regression_results.png` | Combined visualization of top GD results, OLS, and best fit |
| Abalone | Polynomial Regression | `plots/abalone_regression_results.png` | Per-feature polynomial fits with selected degrees |

---

## 🧩 Streamlit Dashboard

A basic interactive dashboard for exploring the regression models.

To launch:

```bash
streamlit run app.py
```

**Features:**

- **GDP Tab:** Interactive parameter adjustment (learning rate, epochs) with single regression line visualization
- **Abalone Tab:** Adjustable max polynomial degree with external plot generation

**Note:** For complete model comparisons and detailed visualizations, run the individual Python scripts directly.

---

## 🧮 Datasets

| Dataset | Description | Source |
|----------|--------------|---------|
| `gdp-vs-happiness.csv` | Global dataset containing 2018 GDP per capita and Happiness scores | [World Happiness Report 2018 / OWID] |
| `training_data.csv` | Abalone measurements used for predicting age (Rings + 1.5 years) | [UCI Machine Learning Repository] |

---

## 📈 Model Details

### GDP vs Happiness — Linear Regression

- **Features:** GDP per capita (PPP, constant 2021$)  
- **Target:** Cantril Ladder Happiness Score (2018)  
- **Optimization:** Gradient Descent (manual η/epoch tuning) and closed-form OLS  
- **Goal:** Identify correlation between economic prosperity and subjective well-being.

### Abalone Dataset — Polynomial Regression

- **Features:** Continuous shell measurements (Length, Diameter, Height, etc.)  
- **Target:** Rings + 1.5 (proxy for age in years)  
- **Degree Selection:** Automatic (1–4) based on training MSE per feature  
- **Method:** Individual polynomial fits per feature with standardized inputs
- **Evaluation Metrics:** Train/Test MSE and per-feature visualization

---

## 🧰 Tools and Technologies

- **Languages:** Python 3.10  
- **Libraries:** NumPy, Pandas, Matplotlib, Streamlit  
- **Environment:** Conda (cross-platform reproducibility)  
- **AI Assistance:** GitHub Copilot and ChatGPT (for code explanation and formatting only)

---

## ⚡ Estimated Carbon Footprint (Development)

Approx. **3.5 kg CO₂**, based on 10 hours of active AI-assisted coding and documentation.

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**S. Pathmanathan**  
5th Year Software Engineering Student @ McMaster University  
Prev @ AMD (Datacenter GPU Validation)  
*Turning ideas into code that actually makes life easier.*
