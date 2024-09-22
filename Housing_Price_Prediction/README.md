Competition by Kaggle  
House Prices - Advanced Regression Techniques:  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

### Project Description: Housing Price Prediction

**Objective:**  
Predict housing prices based on various features of houses using the Housing Price Prediction dataset.

**Steps:**

1. **Data Preparation:**
   - Load data from CSV files for the training set (`train.csv`) and the test set (`test.csv`).
   - Identify categorical and numerical features.

2. **Data Preprocessing:**
   - **Handling Missing Values:**
     - Use `SimpleImputer` for numerical data (fill missing values with the mean).
     - Use `SimpleImputer` for categorical data (fill missing values with the most frequent category).
   - **Encoding Categorical Features:**
     - Apply One-Hot Encoding to convert categorical features into numerical values.
   - **Scaling Data:**
     - Normalize numerical data using `StandardScaler`.

3. **Model Training:**
   - Train a Linear Regression model on the prepared data.

4. **Prediction and Evaluation:**
   - Prepare the test dataset with similar preprocessing steps.
   - Perform predictions on the test dataset.
   - Save results to a CSV file with identifiers and predicted housing prices.

5. **Results:**
   - Generate a CSV file with two columns: `Id` and `SalePrice`, where `SalePrice` contains the predicted housing prices for the test data.

**Tools and Libraries:**
- Python, Pandas, Scikit-learn
