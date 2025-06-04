import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def data_analysis(df):
    print("\nPodsumowanie statystyczne dla danych ilościowych:")
    print(df.describe())

    print("\nKolumny:")
    print(df.columns.tolist())

    print("\nTypy danych:")
    print(df.dtypes)

    print("\nSprawdzenie brakujących danych:")
    print(df.isnull().sum())

    # Wydzielenie kolumn nominalnych i ilościowych
    nominal_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    print(f"\nAtrybuty nominalne ({len(nominal_cols)}): {nominal_cols}")
    print(f"\nAtrybuty ilościowe ({len(numeric_cols)}): {numeric_cols}")

    # Dystrybucja danych ilościowych
    print("\nHistogramy danych ilościowych:")
    df[numeric_cols].hist(figsize=(12, 8), bins=20)
    plt.tight_layout()
    plt.show()

    # Najczęstsze wartości w kolumnach nominalnych
    for col in nominal_cols:
        print(f"\nTop wartości dla {col}:")
        print(df[col].value_counts().head(10))

    # Korelacje między zmiennymi ilościowymi
    print("\nKorelacje między zmiennymi ilościowymi:")
    corr = df[numeric_cols].corr()
    print(corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Macierz korelacji")
    plt.show()


if __name__ == "__main__":
    # read data
    data_path = "vgsales.csv"
    dataframe = pd.read_csv(data_path)
    data_analysis(dataframe)
    # todo
    # Genre - publisher correlations
    # plot sales each year
    # plot sales for each publisher
    # plot sales each platform
    # plot sales each Genre
    # plot sales each platform/year (2D)
    # plot same stuff but / year
    # calc/plot avg sales per game published
    # Group by? Gra wydana na kilka platform ma rekord dla każdej z nich
    # Model przewidujący sprzedaż danego combo Publisher/Platform/Genre ale w następnym roku?
