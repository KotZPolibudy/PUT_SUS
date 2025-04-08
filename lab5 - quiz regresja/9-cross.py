import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import neighbors  # KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# Funkcja rysująca wykres
def draw_and_save_plot(x, y, title, ylabel, filename=None):
    plt.figure(figsize=(8, 6))
    plt.bar(x, y, color='lightgreen' if title == "MAE" else 'salmon')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


# Funkcja do przeprowadzenia kroswalidacji i rysowania wykresów
def cross_validate_and_plot(models, X, y):
    results_kfold = {'Model': [], 'MAE': [], 'RMSE': []}

    for name, model in models.items():
        # Kroswalidacja 10-krotna dla MAE i RMSE
        mae_scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')
        rmse_scores = cross_val_score(model, X, y, cv=10, scoring='neg_root_mean_squared_error')

        # Zbieranie wyników
        results_kfold['Model'].append(name)
        results_kfold['MAE'].append(np.mean(-mae_scores))  # negatywne MAE jest zwracane, więc zamieniamy znak
        results_kfold['RMSE'].append(np.mean(-rmse_scores))  # negatywny RMSE, więc zamieniamy znak

    # Konwersja wyników na DataFrame
    df_kfold = pd.DataFrame(results_kfold)

    # Rysowanie osobnych wykresów
    draw_and_save_plot(df_kfold['Model'], df_kfold['MAE'], 'MAE (Kroswalidacja)', 'MAE', 'wykresy/mae_kroswalidacja.png')
    draw_and_save_plot(df_kfold['Model'], df_kfold['RMSE'], 'RMSE (Kroswalidacja)', 'RMSE', 'wykresy/rmse_kroswalidacja.png')

    return df_kfold


# Wczytanie danych
dane = pd.read_csv("151879-regression.txt", sep="\t")
Xregr = dane.iloc[:, 0:-1].values
yregr = dane.iloc[:, -1].values

# Definicja regresorów
regressors = {
    'Linear': linear_model.LinearRegression(),
    'KNeighbors': neighbors.KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(max_depth=2),
    'MLP': MLPRegressor(max_iter=1000),
    'SVR Linear': SVR(kernel='linear'),
    'SVR RBF': SVR(kernel='rbf'),
}

regressors_normalized = {
    'Linear': make_pipeline(StandardScaler(), linear_model.LinearRegression()),
    'KNeighbors': make_pipeline(StandardScaler(), neighbors.KNeighborsRegressor()),
    'Decision Tree': make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=2)),
    'MLP': make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000)),
    'SVR Linear': make_pipeline(StandardScaler(), SVR(kernel='linear')),
    'SVR RBF': make_pipeline(StandardScaler(), SVR(kernel='rbf'))
}

# Przeprowadzenie kroswalidacji dla modeli z normalizacją
df_kfold_results = cross_validate_and_plot(regressors_normalized, Xregr, yregr)

# Dla porównania: przeprowadzamy kroswalidację także dla modeli bez normalizacji
df_kfold_results_no_normalization = cross_validate_and_plot(regressors, Xregr, yregr)
