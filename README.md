# 🏠 Linear Regression: House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Internship](https://img.shields.io/badge/Project-Internship%20Task%201-success.svg)]()

A machine learning model that predicts house prices based on property features using Linear Regression. This is the first internship task demonstrating fundamental ML concepts with Pandas, Python, and Scikit-learn.

## 📊 Overview

This project implements a **Linear Regression** model to predict house prices using three key features:
- Square footage of the property
- Number of bedrooms
- Number of bathrooms

The model learns the relationship between these features and house prices, enabling accurate predictions for new properties.

## 📋 Table of Contents
- [Project Details](#project-details)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Implementation](#implementation)
- [Results](#results)
- [How to Use](#how-to-use)
- [Technologies Used](#technologies-used)

## 🎯 Project Details

**Objective**: Build a linear regression model to accurately predict house prices

**Internship**: SCI Internship Task 1

**Difficulty**: Beginner-Friendly

**Concepts Covered**:
- Data preprocessing and exploration
- Feature scaling and normalization
- Model training and evaluation
- Performance metrics (MSE, R², MAE)
- Prediction on new data

## ✨ Features

- ✅ **Data Preprocessing**: Clean and prepare real estate data
- ✅ **Feature Engineering**: Extract and scale relevant features
- ✅ **Model Training**: Implement linear regression using Scikit-learn
- ✅ **Model Evaluation**: Comprehensive performance metrics
- ✅ **Visualization**: Plot predictions vs actual values
- ✅ **Predictions**: Make predictions on new properties
- ✅ **Jupyter Notebook**: Complete workflow in interactive notebook

## 📈 Dataset

The dataset includes:
- **Square Footage** (X₁): Property size in square feet
- **Bedrooms** (X₂): Number of bedrooms
- **Bathrooms** (X₃): Number of bathrooms
- **Price** (Y): Target variable - house price in dollars

### Dataset Statistics:
```
Total samples: Multiple property records
Features: 3 numerical features
Target variable: House Price
Data type: Numerical
```

## 📦 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- pip

### Steps

```bash
# Clone repository
git clone https://github.com/hasinigovardhan2413-sys/SCT_ML_1.git
cd SCT_ML_1

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## 🔧 Implementation

### 1. Data Loading and Exploration

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
df = pd.read_csv('house_prices.csv')

# Explore data
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
```

### 2. Data Preprocessing

```python
# Handle missing values
df = df.dropna()

# Separate features and target
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']

# Check for outliers
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### 3. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Scale features (important for better model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully!")
```

### 4. Model Training

```python
# Create and train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Get model parameters
print("Model Coefficients:")
print(f"  Square Footage: {model.coef_[0]:.4f}")
print(f"  Bedrooms: {model.coef_[1]:.4f}")
print(f"  Bathrooms: {model.coef_[2]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
```

### 5. Model Evaluation

```python
# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate performance metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("Model Performance:")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing R² Score: {test_r2:.4f}")
print(f"Mean Squared Error: {test_mse:.2f}")
print(f"Root Mean Squared Error: ${test_rmse:,.2f}")
print(f"Mean Absolute Error: ${test_mae:,.2f}")
```

### 6. Visualization

```python
import matplotlib.pyplot as plt

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot residuals
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_test, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.show()
```

## 📊 Results

### Model Performance Metrics:

| Metric | Description | Value |
|--------|-------------|-------|
| **R² Score** | Proportion of variance explained (0-1) | [Value] |
| **RMSE** | Root mean squared error in dollars | [Value] |
| **MAE** | Mean absolute error in dollars | [Value] |
| **MSE** | Mean squared error | [Value] |

### Key Insights:
- Linear relationship between features and house prices is significant
- Model explains [X]% of variance in house prices
- Average prediction error is approximately $[Value]
- Feature scaling improves model convergence

### Model Equation:

```
Price = β₀ + β₁(SquareFootage) + β₂(Bedrooms) + β₃(Bathrooms)

Where:
  β₀ = Intercept (base price)
  β₁ = Price per square foot
  β₂ = Price premium per bedroom
  β₃ = Price premium per bathroom
```

## 🎮 How to Use

### Run the Jupyter Notebook

```bash
jupyter notebook House_Price_Prediction.ipynb
```

### Make Predictions on New Data

```python
# New property data
new_property = [[2500, 4, 2.5]]  # 2500 sq ft, 4 bed, 2.5 bath

# Scale the new data using the same scaler
new_scaled = scaler.transform(new_property)

# Predict price
predicted_price = model.predict(new_scaled)
print(f"Predicted price: ${predicted_price[0]:,.2f}")

# Batch predictions
new_properties = [
    [2000, 3, 2],
    [3000, 4, 3],
    [1500, 2, 1.5]
]

new_scaled = scaler.transform(new_properties)
predictions = model.predict(new_scaled)

for i, pred in enumerate(predictions):
    print(f"Property {i+1}: ${pred:,.2f}")
```

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Machine learning library |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical graphics (optional) |
| **Jupyter** | Interactive notebook environment |
| **Google Colab** | Cloud-based notebook (optional) |

## 📚 Learning Concepts

This project covers:
- ✅ Linear Regression fundamentals
- ✅ Ordinary Least Squares (OLS) method
- ✅ Data normalization and feature scaling
- ✅ Train-test split methodology
- ✅ Regression performance metrics
- ✅ Model validation techniques
- ✅ Overfitting and underfitting concepts
- ✅ Data visualization best practices

## 🔍 Mathematical Background

### Linear Regression Formula:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Error (Residual) = y - ŷ

Cost Function (Mean Squared Error):
MSE = (1/n) Σ(y - ŷ)²

Objective: Minimize MSE to find optimal β values
```

## 💾 Files

```
SCT_ML_1/
├── House_Price_Prediction.ipynb  # Main analysis notebook
├── house_prices.csv              # Dataset
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🚀 Future Enhancements

- [ ] Implement polynomial regression for comparison
- [ ] Try Ridge/Lasso regression for regularization
- [ ] Add more features (age, location, condition, etc.)
- [ ] Implement cross-validation
- [ ] Feature importance analysis
- [ ] Compare with other algorithms (Random Forest, XGBoost)
- [ ] Deploy as REST API
- [ ] Create web interface for predictions

## ⚡ Advanced Topics to Explore

- **Regularization**: Ridge and Lasso regression
- **Feature Selection**: Identifying most important features
- **Cross-Validation**: K-fold validation
- **Ensemble Methods**: Combining multiple models
- **Non-Linear Relationships**: Polynomial and interaction terms

## 🤝 Contributing

Found a bug or have suggestions? Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

MIT License - Feel free to use this project for learning and educational purposes.

## 📞 Contact

**Hasini Govardhan**
- 🔗 GitHub: [@hasinigovardhan2413-sys](https://github.com/hasinigovardhan2413-sys)
- 📧 Email: hasinigovardhan2413@gmail.com
- 💼 Open for internships and collaborations!

---

⭐ **If you found this project helpful, please star it!** It helps others discover it and supports my learning journey.

**Happy Learning! This is my first step in the ML journey! 🚀**
