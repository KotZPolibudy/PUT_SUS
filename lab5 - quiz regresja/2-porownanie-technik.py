import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import neighbors  # KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os


# Funkcja obliczająca i wyświetlająca metryki
def print_metrics(regressor, description, X, y):
    y_pred = regressor.predict(X)
    print(f"{description}:")
    print(f"  R^2 = {r2_score(y, y_pred):.2f}")
    print(f"  MAE = {mean_absolute_error(y, y_pred):.1f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y, y_pred)):.1f}")
    print()


# Wczytanie danych
dane = pd.read_csv("151879-regression.txt", sep="\t")

# Indeksowanie danych (korzystając z Pandas)
Xregr = dane.iloc[:, 0:-1].values  # Wszystkie kolumny oprócz ostatniej
yregr = dane.iloc[:, -1].values    # Ostatnia kolumna

regressors = {
    'Linear Regression': linear_model.LinearRegression(),
    'KNeighbors Regressor': neighbors.KNeighborsRegressor(),
    'Decision Tree Regressor (max_depth=2)': DecisionTreeRegressor(max_depth=2),
    'MLP Regressor': MLPRegressor(max_iter=1000),
    'SVR Linear': SVR(kernel='linear'),
    'SVR RBF': SVR(kernel='rbf')
}
results = {
    'Model': [],
    'R^2': [],
    'MAE': [],
    'RMSE': []
}

for name, model in regressors.items():
    model.fit(Xregr, yregr)
    y_pred = model.predict(Xregr)
    results['Model'].append(name)
    results['R^2'].append(r2_score(yregr, y_pred))
    results['MAE'].append(mean_absolute_error(yregr, y_pred))
    results['RMSE'].append(np.sqrt(mean_squared_error(yregr, y_pred)))

# Tworzenie DataFrame z wynikami
df_results = pd.DataFrame(results)

# Wykresy
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R^2
axes[0].bar(df_results['Model'], df_results['R^2'], color='skyblue')
axes[0].set_title("R^2")
axes[0].set_ylabel("R^2")

# MAE
axes[1].bar(df_results['Model'], df_results['MAE'], color='lightgreen')
axes[1].set_title("MAE")
axes[1].set_ylabel("MAE")

# RMSE
axes[2].bar(df_results['Model'], df_results['RMSE'], color='salmon')
axes[2].set_title("RMSE")
axes[2].set_ylabel("RMSE")

plt.tight_layout()
zapisz_do = "wykresy/regresje.png"
os.makedirs(os.path.dirname(zapisz_do), exist_ok=True)
plt.savefig(zapisz_do)

# Wypisanie wyników dla wszystkich modeli
for name, model in regressors.items():
    print_metrics(model, name, Xregr, yregr)
