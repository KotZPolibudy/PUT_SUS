# Użycie MLP Regressor z normalizacją
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt

dane = pd.read_csv("151879-regression.txt", sep="\t")

Xregr = dane.iloc[:, 0:-1].values
yregr = dane.iloc[:, -1].values

# Tworzenie pipeline z normalizacją i MLP Regressor
mlp_regr = make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000))

# Trenowanie modelu
mlp_regr.fit(Xregr, yregr)

# Przewidywania
y_pred_mlp = mlp_regr.predict(Xregr)

plt.figure(figsize=(8, 6))
plt.scatter(yregr, y_pred_mlp, color='blue', alpha=0.6)
plt.plot([min(yregr), max(yregr)], [min(yregr), max(yregr)], color='red', linestyle='--')
plt.title("Faktyczne wartości vs Przewidywania MLP Regressor")
plt.xlabel("Faktyczne wartości")
plt.ylabel("Przewidywane wartości")
plt.tight_layout()

# Zapisz wykres
plt.savefig("wykresy/mlp_regressor_faktyczne_vs_przewidywane.png")
plt.show()
