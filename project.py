import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("bestsellers with categories.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.histplot(df['User Rating'], bins=20, kde=True)
plt.xlabel("User Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Genre', y='Price', data=df)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.countplot(x='Genre', data=df)
plt.tight_layout()
plt.show()

label_encoder = LabelEncoder()
df['Author_encoded'] = label_encoder.fit_transform(df['Author'])
df['Genre_encoded'] = label_encoder.fit_transform(df['Genre'])

np.random.seed(42)
df['sales'] = (df['Reviews'] * df['User Rating'] * np.random.uniform(0.5, 1.5, len(df))).astype(int)

features = ['Author_encoded', 'Reviews', 'User Rating', 'Year', 'Genre_encoded', 'Price']
target = 'sales'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt) * 100

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr) * 100

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf) * 100

print(mae_dt)
print(rmse_dt)
print(r2_dt)
print(mae_lr)
print(rmse_lr)
print(r2_lr)
print(mae_rf)
print(rmse_rf)
print(r2_rf)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.scatter(y_test, y_pred_dt, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder.fit(df[['Author', 'Genre']])

new_book_input = {
    'Author': 'J.K. Rowling',
    'Genre': 'Fiction',
    'Reviews': 10000,
    'User Rating': 4.9,
    'Year': 2020,
    'Price': 25.00
}

encoded_author = ordinal_encoder.transform([[new_book_input['Author'], new_book_input['Genre']]])[0][0]
encoded_genre = ordinal_encoder.transform([[new_book_input['Author'], new_book_input['Genre']]])[0][1]

new_features = pd.DataFrame([{
    'Author_encoded': encoded_author,
    'Reviews': new_book_input['Reviews'],
    'User Rating': new_book_input['User Rating'],
    'Year': new_book_input['Year'],
    'Genre_encoded': encoded_genre,
    'Price': new_book_input['Price']
}])

new_scaled = scaler.transform(new_features)
predicted_sales = rf_model.predict(new_scaled)

print(int(predicted_sales[0]))

for i in range(60):
    temp_model = DecisionTreeRegressor()
    temp_model.fit(X_train_scaled, y_train)
    temp_preds = temp_model.predict(X_test_scaled)
    temp_mae = mean_absolute_error(y_test, temp_preds)
    temp_rmse = np.sqrt(mean_squared_error(y_test, temp_preds))
    temp_r2 = r2_score(y_test, temp_preds)
    df_mean = df.mean(numeric_only=True)
    df_median = df.median(numeric_only=True)
    df_max = df.max(numeric_only=True)
    df_min = df.min(numeric_only=True)
    genre_group = df.groupby('Genre').agg({"Price": ["mean", "max"], "Reviews": "sum"})
    author_group = df.groupby('Author').agg({"User Rating": "mean", "Reviews": "mean"})
    correlation_matrix = df.corr(numeric_only=True)

print("End of program.")

