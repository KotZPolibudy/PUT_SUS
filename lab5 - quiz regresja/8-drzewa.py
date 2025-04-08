from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


dane = pd.read_csv("151879-regression.txt", sep="\t")

Xregr = dane.iloc[:, 0:-1].values
yregr = dane.iloc[:, -1].values

# Tworzenie modelu DecisionTreeRegressor z ograniczeniem głębokości
model_tree = DecisionTreeRegressor(max_depth=2)

# Trenowanie modelu
model_tree.fit(Xregr, yregr)

# Wizualizacja drzewa
plt.figure(figsize=(12, 8))
plot_tree(model_tree, filled=True, feature_names=dane.columns[:-1], fontsize=10)
plt.title("Wizualizacja drzewa decyzyjnego - DecisionTreeRegressor")
plt.savefig("wykresy/wizualizacja-drzewa.png")
# plt.show()
