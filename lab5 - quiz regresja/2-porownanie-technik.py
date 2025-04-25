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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Funkcja obliczająca i wyświetlająca metryki
def print_metrics(regressor, description, X, y):
    y_pred = regressor.predict(X)
    print(f"{description}:")
    print(f"  R^2 = {r2_score(y, y_pred):.2f}")
    print(f"  MAE = {mean_absolute_error(y, y_pred):.1f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y, y_pred)):.1f}")
    print()


# Funkcja rysująca wykres
def draw_and_save_plot(x, y, title, ylabel, filename=None):
    plt.figure(figsize=(8, 6))
    plt.bar(x, y, color='skyblue' if title == "R^2" else ('lightgreen' if title == "MAE" else 'salmon'))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


# Wczytanie danych
dane = pd.read_csv("151879-regression.txt", sep="\t")

Xregr = dane.iloc[:, 0:-1].values
yregr = dane.iloc[:, -1].values

regressors = {
    'Linear': linear_model.LinearRegression(),
    'KNeighbors': neighbors.KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(max_depth=2),
    'MLP': MLPRegressor(max_iter=1000),
    'SVR Linear': SVR(kernel='linear'),
    'SVR RBF': SVR(kernel='rbf'),
    'Full Decision Tree': DecisionTreeRegressor()
}
results = {
    'Model': [],
    'R^2': [],
    'MAE': [],
    'RMSE': []
}

regressors_normalized = {
    'Linear': make_pipeline(StandardScaler(), linear_model.LinearRegression()),
    'KNeighbors': make_pipeline(StandardScaler(), neighbors.KNeighborsRegressor()),
    'Decision Tree': make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=2)),
    'MLP': make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000)),
    'SVR Linear': make_pipeline(StandardScaler(), SVR(kernel='linear')),
    'SVR RBF': make_pipeline(StandardScaler(), SVR(kernel='rbf'))
}

results_normalized = {
    'Model': [], 'R^2': [], 'MAE': [], 'RMSE': []
}
"""
for name, model in regressors.items():
    model.fit(Xregr, yregr)
    y_pred = model.predict(Xregr)
    results['Model'].append(name)
    results['R^2'].append(r2_score(yregr, y_pred))
    results['MAE'].append(mean_absolute_error(yregr, y_pred))
    results['RMSE'].append(np.sqrt(mean_squared_error(yregr, y_pred)))


df_results = pd.DataFrame(results)
draw_and_save_plot(df_results['Model'], df_results['R^2'], "R^2", "R^2", "wykresy/regresja_r2.png")
draw_and_save_plot(df_results['Model'], df_results['MAE'], "MAE", "MAE", "wykresy/regresja_mae.png")
draw_and_save_plot(df_results['Model'], df_results['RMSE'], "RMSE", "RMSE", "wykresy/regresja_rmse.png")

# Wypisanie wyników dla wszystkich modeli
for name, model in regressors.items():
    print_metrics(model, name, Xregr, yregr)
"""

for name, model in regressors_normalized.items():
    model.fit(Xregr, yregr)
    y_pred = model.predict(Xregr)
    results_normalized['Model'].append(name)
    results_normalized['R^2'].append(r2_score(yregr, y_pred))
    results_normalized['MAE'].append(mean_absolute_error(yregr, y_pred))
    results_normalized['RMSE'].append(np.sqrt(mean_squared_error(yregr, y_pred)))


df_results_normalized = pd.DataFrame(results_normalized)
draw_and_save_plot(df_results_normalized['Model'], df_results_normalized['R^2'], "R^2 (Normalizacja)", "R^2", "wykresy/regresja_r2_normalized.png")
draw_and_save_plot(df_results_normalized['Model'], df_results_normalized['MAE'], "MAE (Normalizacja)", "MAE", "wykresy/regresja_mae_normalized.png")
draw_and_save_plot(df_results_normalized['Model'], df_results_normalized['RMSE'], "RMSE (Normalizacja)", "RMSE", "wykresy/regresja_rmse_normalized.png")

# Wypisanie wyników dla wszystkich modeli
for name, model in regressors_normalized.items():
    print_metrics(model, name, Xregr, yregr)
