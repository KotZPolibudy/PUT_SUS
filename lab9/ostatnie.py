import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay


def prepare_X_y(df):
    X = df.drop(columns='class')  # Zmienna docelowa musi się nazywać 'target'
    y = df['class']
    return X.values, y.values


def plot_avg_confusion_matrix_for_adaboost(df):
    X, y = prepare_X_y(df)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Parametry AdaBoosta
    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    adaboost = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=150,
        learning_rate=1.0,
        random_state=42
    )

    all_cm = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone(adaboost)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        all_cm.append(cm)

    all_cm = np.array(all_cm)
    mean_cm = np.mean(all_cm, axis=0)
    std_cm = np.std(all_cm, axis=0)

    return all_cm, mean_cm, std_cm


def plot_avg_confusion_matrix(df):
    # Załaduj dane
    X, y = prepare_X_y(df)

    # Twój najlepszy model Voting_B (przykład)
    model = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, max_iter=500, random_state=42)),
            ('gnb', GaussianNB()),
            ('dt', DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42))
        ],
        voting='soft'
    )
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_cm = []

    for train_idx, test_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        cm = confusion_matrix(y[test_idx], y_pred, labels=np.unique(y))
        all_cm.append(cm)

    avg_cm = np.mean(all_cm, axis=0).astype(int)
    mean_cm = np.mean(all_cm, axis=0)
    std_cm = np.std(all_cm, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm, display_labels=np.unique(y))
    disp.plot(cmap="Blues")
    plt.title("Średnia macierz pomyłek (10-fold CV)")
    plt.tight_layout()
    plt.savefig("avg_confusion_matrix.png")
    # plt.show()
    return all_cm, mean_cm, std_cm

def read_data(data_path):
    """Wczytuje dane z pliku."""
    return pd.read_csv(data_path, sep="\t")

if __name__ == "__main__":
    data_path = "151879-ensembles.txt"
    df = read_data(data_path)
    all_cm, mean_cm, std_cm = plot_avg_confusion_matrix(df)
    print("Średnia macierz pomyłek:", np.round(mean_cm, 2))
    print("Odchylenie standardowe:", np.round(std_cm, 2))

    accuracies = []
    gmeans = []

    # Zakładamy, że y_true i y_pred z każdego folda są dostępne lub można je uzyskać
    # Jeśli nie, to trzeba je zebrać w funkcji `plot_avg_confusion_matrix` i zwrócić dodatkowo

    for cm in all_cm:
        correct = np.trace(cm)
        total = np.sum(cm)
        acc = correct / total
        accuracies.append(acc)

        # Obliczenie G-mean z macierzy pomyłek
        # Zakładamy binary classification: [TN, FP], [FN, TP]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = (sensitivity * specificity) ** 0.5
            gmeans.append(gmean)
        else:
            gmeans.append(np.nan)  # lub można pominąć dla nie-binarnych

    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)
    gmean_mean = np.nanmean(gmeans)
    gmean_std = np.nanstd(gmeans)

    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"G-mean:   {gmean_mean:.4f} ± {gmean_std:.4f}")
