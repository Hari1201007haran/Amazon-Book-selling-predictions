# 📘 Amazon Book Selling Predictions

Predicts Amazon book sales using historical bestseller data. Applies data preprocessing, label encoding, and a Decision Tree Regressor to estimate sales. Includes visualization, feature engineering, and new book prediction using encoded attributes.

## 📁 Dataset

- **File**: `bestsellers with categories.csv`
- **Attributes**: Name, Author, User Rating, Reviews, Price, Year, Genre

## ⚙️ Features

- Data cleaning and visualization
- Label encoding (`Author`, `Genre`)
- Feature scaling with `StandardScaler`
- Synthetic `sales` column based on rating and reviews
- Model training using `DecisionTreeRegressor`
- Evaluation metrics: MAE, RMSE, R²
- Sales prediction for new book entries

## 🛠 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

## 📊 Sample Output


Decision Tree MAE: 53106.7
Decision Tree RMSE: 77676.9
Decision Tree R2: 67.48
Predicted Sales for the new book: 143892


## 👤 Author

**Hariharan**  
Project: *Amazon Book Selling Predictions*
