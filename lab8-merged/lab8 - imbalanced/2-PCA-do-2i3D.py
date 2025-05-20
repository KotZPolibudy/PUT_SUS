import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # wymagane do 3D wykresów

# Pomocnicza funkcja: dodawanie jittera
def add_jitter(arr, noise=0.02):
    return arr + np.random.normal(0, noise, arr.shape)

# PCA 2D
def pca_2d_wykres(X, y, zapisz_do=None, jitter=False, alpha=1.0, minority_last=False):

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    wariancja = np.sum(pca.explained_variance_ratio_) * 100

    unique_classes, counts = np.unique(y, return_counts=True)
    class_with_counts = list(zip(unique_classes, counts))
    # Jeśli minority_last, sortuj malejąco po liczności – mniejszościowe klasy będą rysowane później
    if minority_last:
        class_order = [cls for cls, _ in sorted(class_with_counts, key=lambda x: x[1], reverse=True)]
    else:
        class_order = [cls for cls, _ in class_with_counts]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    suffix = ""
    if jitter:
        suffix += "_jitter"
    if alpha < 1.0:
        suffix += "_alpha"

    if zapisz_do:
        nazwa_pliku = zapisz_do.replace(".png", f"{suffix}.png")
    else:
        nazwa_pliku = None

    plt.figure(figsize=(8, 6))
    for idx, cls in enumerate(class_order):
        indices = np.where(y == cls)
        punkty = X_pca[indices]
        if jitter:
            punkty = add_jitter(punkty)

        plt.scatter(punkty[:, 0], punkty[:, 1],
                    alpha=alpha,
                    label=f"Klasa {cls}",
                    color=colors[idx % len(colors)])
    plt.title(f"PCA 2D (wariancja: {wariancja:.2f}%)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if nazwa_pliku:
        os.makedirs(os.path.dirname(nazwa_pliku), exist_ok=True)
        plt.savefig(nazwa_pliku)
        plt.close()
    else:
        plt.show()

    return wariancja

# PCA 3D
def pca_3d_wykres(X, y, zapisz_do=None, jitter=False, alpha=1.0, minority_last=False):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    wariancja = np.sum(pca.explained_variance_ratio_) * 100

    unique_classes, counts = np.unique(y, return_counts=True)
    class_with_counts = list(zip(unique_classes, counts))
    # Jeśli minority_last, sortuj malejąco po liczności – mniejszościowe klasy będą rysowane później
    if minority_last:
        class_order = [cls for cls, _ in sorted(class_with_counts, key=lambda x: x[1], reverse=True)]
    else:
        class_order = [cls for cls, _ in sorted(class_with_counts, key=lambda x: x[1])]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    suffix = ""
    if jitter:
        suffix += "_jitter"
    if alpha < 1.0:
        suffix += "_alpha"

    if zapisz_do:
        nazwa_pliku = zapisz_do.replace(".png", f"{suffix}.png")
    else:
        nazwa_pliku = None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for idx, cls in enumerate(class_order):
        indices = np.where(y == cls)
        punkty = X_pca[indices]
        if jitter:
            punkty = add_jitter(punkty)

        ax.scatter(punkty[:, 0], punkty[:, 1], punkty[:, 2],
                   alpha=alpha,
                   label=f"Klasa {cls}",
                   color=colors[idx % len(colors)])
    ax.set_title(f"PCA 3D (wariancja: {wariancja:.2f}%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()

    if nazwa_pliku:
        os.makedirs(os.path.dirname(nazwa_pliku), exist_ok=True)
        plt.savefig(nazwa_pliku)
        plt.close()
    else:
        plt.show()

    return wariancja

# Wczytanie danych
df = pd.read_csv("151879-imbalanced.txt", sep="\t")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Generowanie wykresów 2D
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d.png", jitter=False, alpha=1.0)
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d.png", jitter=True, alpha=1.0)
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d.png", jitter=False, alpha=0.5)
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d.png", jitter=True, alpha=0.5)

# Generowanie wykresów 3D
pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d.png", jitter=False, alpha=1.0)
pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d.png", jitter=True, alpha=1.0)
pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d.png", jitter=False, alpha=0.5)
pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d.png", jitter=True, alpha=0.5)

# Generowanie wykresów z sortowaniem
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d_minority-last.png", jitter=False, alpha=1.0, minority_last=True)
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d_minority-last.png", jitter=True, alpha=1.0, minority_last=True)
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d_minority-last.png", jitter=False, alpha=0.5, minority_last=True)
pca_2d_wykres(X, y, zapisz_do="wyniki/pca_2d_minority-last.png", jitter=True, alpha=0.5, minority_last=True)

pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d_minority-last.png", jitter=False, alpha=1.0, minority_last=True)
pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d_minority-last.png", jitter=True, alpha=1.0, minority_last=True)
pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d_minority-last.png", jitter=False, alpha=0.5, minority_last=True)
pca_3d_wykres(X, y, zapisz_do="wyniki/pca_3d_minority-last.png", jitter=True, alpha=0.5, minority_last=True)
