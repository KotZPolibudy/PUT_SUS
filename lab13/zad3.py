from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np

# Wczytanie danych
dane = pd.read_csv("151879-regression.txt", sep="\t")
X = dane.iloc[:, :-1].values
y = dane.iloc[:, -1].values
feature_names = dane.columns[:-1].tolist()

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu regresyjnego drzewa decyzyjnego
model = DecisionTreeRegressor(max_depth=2, random_state=42)
model.fit(X_train, y_train)

# Tworzenie explainer-a LIME
explainer = LimeTabularExplainer(X_train, feature_names=feature_names, verbose=True, mode='regression')

# Wybranie kilku przypadków testowych do wyjaśnienia
test_indices = [0, 1, 5]
lime_explanations = []
for i in test_indices:
    exp = explainer.explain_instance(X_test[i], model.predict, num_features=4)
    lime_explanations.append((i, exp.as_list()))

print(lime_explanations)
