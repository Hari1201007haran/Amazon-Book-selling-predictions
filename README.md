# 📘 Amazon Book Selling Predictions

Predicts Amazon book sales using historical bestseller data. Applies preprocessing, label encoding, feature scaling, and multiple regression models (Decision Tree, Linear, Random Forest) to estimate synthetic sales values. Includes visualizations and predictions for new book entries.

## 📁 Dataset

- **File**: `bestsellers with categories.csv`
- **Attributes**: Name, Author, User Rating, Reviews, Price, Year, Genre

## ⚙️ Features

- Data inspection and visualization (histograms, box plots, scatter plots)
- Label encoding (`Author`, `Genre`) and ordinal encoding for new inputs
- Feature scaling with `StandardScaler`
- Synthetic `sales` column combining `Reviews`, `User Rating`, and randomness
- Model training using:
  - `DecisionTreeRegressor`
  - `LinearRegression`
  - `RandomForestRegressor`
- Performance evaluation: MAE, RMSE, R²
- Sales prediction for a new book (`Author: J.K. Rowling`, `Genre: Fiction`, etc.)
- Repeated model training for robustness check
- Statistical summaries and correlation analysis

## 🛠 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

## 📊 Sample Output

Decision Tree MAE: 54872.4
Decision Tree RMSE: 78124.2
Decision Tree R2: 66.25

Linear Regression MAE: 48621.1
Linear Regression RMSE: 70234.3
Linear Regression R2: 71.52

Random Forest MAE: 46782.9
Random Forest RMSE: 68891.5
Random Forest R2: 72.74

Predicted Sales for the new book: 152043


## 👤 Author

**Hariharan**  
Project: *Amazon Book Selling Predictions*
