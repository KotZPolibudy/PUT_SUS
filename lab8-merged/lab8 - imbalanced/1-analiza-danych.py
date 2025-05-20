import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def skrzypcowy(data, zapisz_do=None):
    plt.figure(figsize=(20, 6))
    sns.violinplot(data=data, inner="box")
    plt.xticks(rotation=45)
    plt.title("Rozkład wartości atrybutów (skrzypcowy)")
    plt.tight_layout()
    if zapisz_do:
        os.makedirs(os.path.dirname(zapisz_do), exist_ok=True)
        plt.savefig(zapisz_do)
    else:
        plt.show()


def pudelkowy(data, zapisz_do=None):
    plt.figure(figsize=(20, 6))
    sns.boxplot(data=data)
    plt.xticks(rotation=45)
    plt.title("Rozkład wartości atrybutów (boxplot)")
    plt.tight_layout()
    if zapisz_do:
        os.makedirs(os.path.dirname(zapisz_do), exist_ok=True)
        plt.savefig(zapisz_do)
    else:
        plt.show()


def heatmapa_korelacji(data, zapisz_do=None):
    plt.figure(figsize=(16, 10))
    korelacje = data.corr()
    sns.heatmap(korelacje, cmap="coolwarm", annot=False)
    plt.title("Macierz korelacji cech")
    plt.tight_layout()
    if zapisz_do:
        os.makedirs(os.path.dirname(zapisz_do), exist_ok=True)
        plt.savefig(zapisz_do)
    else:
        plt.show()


# działa nawet lepiej niż polecenie podane z numpy, bo ogarnia nagłówki i daje ładny print ;)
df = pd.read_csv("151879-imbalanced.txt", sep="\t")

print("Kształt danych:", df.shape)
print("Pierwsze 5 wierszy:\n", df.head())

# Podział na cechy i etykietę (ostatnia kolumna jako y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
ALL = df.iloc[:, :]

print("\nLiczba przykładów:", X.shape[0])
print("Liczba atrybutów:", X.shape[1])
print("Atrybuty klasy:", 1)

print("\nLiczność klas:")
print(y.value_counts())

# Statystyki atrybutów
print("\nStatystyki atrybutów:")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.describe(include='all').T)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.width')

# Wykres skrzypcowy
skrzypcowy(X, "wykresy/skrzypcowy.png")
pudelkowy(X, "wykresy/boxplot_atrybuty.png")
heatmapa_korelacji(X, "wykresy/korelacje.png")
skrzypcowy(ALL, "wykresy/skrzypcowyALL.png")
