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

def read_data(data_path):
    """Wczytuje dane z pliku."""
    return pd.read_csv(data_path, sep="\t")

def summarize_data(df):
    """Wyświetla podstawowe informacje o danych."""
    print("📏 Kształt danych:", df.shape)
    print("📊 Typy danych:\n", df.dtypes)

def analyze_class_distribution(df):
    """Wyświetla rozkład klas."""
    class_col = df.columns[-1]
    print("\n🔢 Proporcje klas:")
    print(df[class_col].value_counts(normalize=True))

def describe_features(df):
    """Wyświetla statystyki opisowe."""
    print("\n📈 Statystyki opisowe:")
    print(df.describe(include='all'))

def plot_violinplot(df):
    """Tworzy wykres skrzypcowy dla cech numerycznych."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns[:-1]  # pomijamy klasę
    melted = df[numeric_cols].melt(var_name="Atrybut", value_name="Wartość")

    plt.figure(figsize=(20, 8))
    sns.violinplot(data=melted, x="Atrybut", y="Wartość", inner="quartile", density_norm="width", cut=0)
    plt.xticks(rotation=90)
    plt.title("Rozkład wartości wszystkich atrybutów")
    plt.tight_layout()
    plt.savefig("rozklad_wartosci.png")
    # plt.show()

def plot_correlation_heatmap(df):
    # Wybieramy tylko kolumny numeryczne, bez etykiety klasy
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:-1]

    # Obliczanie macierzy korelacji
    corr_matrix = df[numeric_cols].corr()

    # Rysowanie heatmapy
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji pomiędzy atrybutami")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("macierz_korelacji.png")
    #plt.show()


def data_analysis(df):
    summarize_data(df)
    analyze_class_distribution(df)
    describe_features(df)
    plot_violinplot(df)
    plot_correlation_heatmap(df)

def visualize_pca(df):
    X, y = prepare_X_y(df)
    
    # Zakodowanie klasy jeśli jest tekstowa
    y_encoded = LabelEncoder().fit_transform(y)

    # PCA 2D
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    var_2d = np.sum(pca_2d.explained_variance_ratio_) * 100

    # PCA 3D
    pca_3d = PCA(n_components=3)
    X_3d = pca_3d.fit_transform(X)
    var_3d = np.sum(pca_3d.explained_variance_ratio_) * 100

    # Kolory i etykiety
    unique_classes, counts = np.unique(y, return_counts=True)
    palette = sns.color_palette("Set1", n_colors=len(unique_classes))

    # Znajdź klasę większościową (max próbek)
    majority_class = unique_classes[np.argmax(counts)]
    minority_classes = [cls for cls in unique_classes if cls != majority_class]

    # --- Wykres 2D ---
    plt.figure(figsize=(8, 6))

    # Najpierw rysujemy klasę większościową
    mask_majority = y == majority_class
    plt.scatter(
        X_2d[mask_majority, 0], X_2d[mask_majority, 1],
        label=f"Klasa {majority_class} (większość)",
        alpha=0.25 if np.sum(mask_majority) > 1000 else 1.0,
        s=20 if np.sum(mask_majority) > 1000 else 50,
        edgecolor='k',
        color=palette[np.where(unique_classes == majority_class)[0][0]]
    )

    # Potem rysujemy pozostałe klasy (mniejszościowe) na wierzchu
    for label in minority_classes:
        mask = y == label
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=f"Klasa {label} (mniejszość)",
            alpha=1.0,
            s=50,
            edgecolor='k',
            color=palette[np.where(unique_classes == label)[0][0]]
        )
    plt.title(f"PCA - Projekcja 2D (zachowana wariancja: {var_2d:.2f}%)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_2d.png")
    # plt.show()

    # --- Wykres 3D ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i, label in enumerate(unique_classes):
        mask = y == label
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                   label=str(label), alpha=0.6,
                   s=40 if np.sum(mask) > 50 else 80, color=palette[i])
    ax.set_title(f"PCA - Projekcja 3D (zachowana wariancja: {var_3d:.2f}%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_3d.png")
    # plt.show()

def evaluate_ensemble_methods(df):    
    X, y = prepare_X_y(df)
    
    # Przygotowanie CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    def evaluate_model(model):
        scores = []
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            score = geometric_mean_score(y[test_idx], y_pred)
            scores.append(score)
        return np.mean(scores), np.std(scores)

    results = {}
    
    # 1. RandomForestClassifier
    print(f"\nRandomForestClassifier:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    print(f"Model: {rf}")
    results['RandomForest'] = evaluate_model(rf)
    
    # 2. AdaBoost z dwoma bazowymi klasyfikatorami
    print(f"\nAdaBoost z dwoma bazowymi klasyfikatorami:")
    ada1 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
    ada2 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=42)
    print(f"Klasyfikator 1: {ada1.estimator}")
    print(f"Klasyfikator 2: {ada2.estimator}")
    results['AdaBoost_depth1'] = evaluate_model(ada1)
    results['AdaBoost_depth3'] = evaluate_model(ada2)
    
    # 3. VotingClassifier z dwoma zestawami bazowych klasyfikatorów
    print(f"\nVotingClassifier z dwoma zestawami bazowych klasyfikatorów:")
    voting_estimators_A = [
        ('dt', DecisionTreeClassifier(max_depth=3)),
        ('svc', SVC(probability=True, random_state=42)),
        ('gnb', GaussianNB())
    ]
    voting_estimators_B = [
        ('mlp', MLPClassifier(max_iter=500, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('qda', QuadraticDiscriminantAnalysis())
    ]
    print(f"Zestaw A: {voting_estimators_A}")
    print(f"Zestaw B: {voting_estimators_B}")
    voting_A = VotingClassifier(estimators=voting_estimators_A, voting='soft')
    voting_B = VotingClassifier(estimators=voting_estimators_B, voting='soft')
    results['Voting_A'] = evaluate_model(voting_A)
    results['Voting_B'] = evaluate_model(voting_B)
    
    # 4. StackingClassifier z dwoma zestawami bazowych klasyfikatorów i różnymi finalnymi klasyfikatorami
    print(f"\nStackingClassifier z dwoma zestawami bazowych klasyfikatorów:")
    stacking_estimators_A = [
        ('dt', DecisionTreeClassifier(max_depth=3)),
        ('svc', SVC(probability=True, random_state=42))
    ]
    stacking_estimators_B = [
        ('mlp', MLPClassifier(max_iter=500, random_state=42)),
        ('gnb', GaussianNB())
    ]
    print(f"Zestaw A: {stacking_estimators_A}")
    print(f"Zestaw B: {stacking_estimators_B}")
    stacking_A = StackingClassifier(estimators=stacking_estimators_A, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
    stacking_B = StackingClassifier(estimators=stacking_estimators_B, final_estimator=RandomForestClassifier(n_estimators=50, random_state=42), cv=5, n_jobs=-1)
    results['Stacking_A'] = evaluate_model(stacking_A)
    results['Stacking_B'] = evaluate_model(stacking_B)
    
    return results

def print_results_zad4(results):
    print("\n📊 Wyniki klasyfikacji:")
    for model, (mean, std) in results.items():
        print(f"{model}: G-mean = {mean:.4f} ± {std:.4f}")

def prepare_X_y(df):
    X = df.select_dtypes(include=['float64', 'int64']).iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def evaluate_models(X, y, normalize=False):
    def get_models():
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
        vote = VotingClassifier(estimators=[
            ('dt', DecisionTreeClassifier(max_depth=5)),
            ('mlp', MLPClassifier(max_iter=500, random_state=42)),
            ('gnb', GaussianNB())
        ], voting='soft')
        stack = StackingClassifier(estimators=[
            ('svc', SVC(probability=True)),
            ('qda', QuadraticDiscriminantAnalysis())
        ], final_estimator=LogisticRegression(max_iter=500))
        return {'RandomForest': rf, 'AdaBoost': ada, 'Voting': vote, 'Stacking': stack}

    results = {}
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = get_models()

    for name, model in models.items():
        print(f"\nOcena modelu: {name}, normalizacja: {normalize}")
        pipeline = make_pipeline(StandardScaler(), model) if normalize else model
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring=make_scorer(geometric_mean_score))
        results[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
    return results

def compare_original_vs_scaled(df):
    X, y = prepare_X_y(df)
    original_results = evaluate_models(X, y, normalize=False)
    scaled_results = evaluate_models(X, y, normalize=True)

    # Przygotowanie danych do wykresu
    labels = list(original_results.keys())
    original_means = [original_results[k]['mean'] for k in labels]
    scaled_means = [scaled_results[k]['mean'] for k in labels]
    diffs = np.array(scaled_means) - np.array(original_means)

    # Wykres różnic
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, diffs, color='skyblue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Różnica G-mean (Znormalizowane - Oryginalne)")
    plt.ylabel("Różnica G-mean")
    plt.tight_layout()

    for bar, diff in zip(bars, diffs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{diff:.3f}", ha='center', va='bottom')

    plt.savefig("gmean_difference.png")
    # plt.show()

    return original_results, scaled_results

def print_results_zad5(original_results, scaled_results):
    print("\n📊 Wyniki porównania oryginalnych i znormalizowanych danych:")
    for (name, original), (_, scaled) in zip(original_results.items(), scaled_results.items()):
        print(f"{name}: Oryginalne G-mean = {original['mean']:.4f} ± {original['std']:.4f}, "
              f"Znormalizowane G-mean = {scaled['mean']:.4f} ± {scaled['std']:.4f}")

def tune_voting_classifier(df):
    X, y = prepare_X_y(df)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    gmean_scorer = make_scorer(geometric_mean_score)

    # Definiujemy bazowe klasyfikatory (wstępna konfiguracja)
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    mlp = MLPClassifier(max_iter=500, random_state=42)
    gnb = GaussianNB()

    voting = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('mlp', mlp), ('gnb', gnb)],
        voting='soft'  # można też testować 'hard'
    )

    # Zakresy parametrów do GridSearch
    param_grid = {
        'dt__max_depth': [5, 10],
        'dt__class_weight': [None, 'balanced'],
        'rf__n_estimators': [100, 150],
        'rf__max_depth': [10, None],
        'rf__class_weight': [None, 'balanced'],
        'mlp__hidden_layer_sizes': [(50,), (100,)],
        'mlp__alpha': [0.0001],
    }

    grid = GridSearchCV(
        estimator=voting,
        param_grid=param_grid,
        scoring=gmean_scorer,
        cv=skf,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X, y)

    print("Najlepszy wynik G-mean:", grid.best_score_)
    print("Najlepsze parametry:", grid.best_params_)

    return grid.best_score_, grid.best_params_

def evaluate_class_weight_effect(df, best_params):
    X, y = prepare_X_y(df)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    gmean_scorer = make_scorer(geometric_mean_score)

    # Tworzymy bazowe klasyfikatory z najlepszymi parametrami (bez class_weight)
    dt_params = {k.replace('dt__', ''): v for k, v in best_params.items() if k.startswith('dt__')}
    rf_params = {k.replace('rf__', ''): v for k, v in best_params.items() if k.startswith('rf__')}
    mlp_params = {k.replace('mlp__', ''): v for k, v in best_params.items() if k.startswith('mlp__')}
    
    # Usuwamy class_weight (na wszelki wypadek, bo chcemy go dodać później)
    dt_params.pop('class_weight', None)
    rf_params.pop('class_weight', None)

    dt = DecisionTreeClassifier(random_state=42, **dt_params)
    rf = RandomForestClassifier(random_state=42, **rf_params)
    mlp = MLPClassifier(max_iter=500, random_state=42, **mlp_params)
    gnb = GaussianNB()

    voting_orig = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('mlp', mlp), ('gnb', gnb)],
        voting='soft'
    )

    # Pipeline oryginalny
    scores_orig = cross_val_score(voting_orig, X, y, cv=skf, scoring=gmean_scorer, n_jobs=-1)
    mean_orig = scores_orig.mean()
    std_orig = scores_orig.std()

    # Teraz tworzymy kopię klasyfikatorów z class_weight='balanced' dla dt i rf
    dt_bal = DecisionTreeClassifier(random_state=42, class_weight='balanced', **dt_params)
    rf_bal = RandomForestClassifier(random_state=42, class_weight='balanced', **rf_params)
    mlp_bal = clone(mlp)
    gnb_bal = GaussianNB()

    voting_bal = VotingClassifier(
        estimators=[('dt', dt_bal), ('rf', rf_bal), ('mlp', mlp_bal), ('gnb', gnb_bal)],
        voting='soft'
    )

    scores_bal = cross_val_score(voting_bal, X, y, cv=skf, scoring=gmean_scorer, n_jobs=-1)
    mean_bal = scores_bal.mean()
    std_bal = scores_bal.std()

    return (mean_orig, std_orig), (mean_bal, std_bal)

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
    

if __name__ == "__main__":
    data_path = "151879-ensembles.txt"
    df = read_data(data_path)
    # zadanie 1
    # data_analysis(df)
    # zadanie 2
    # visualize_pca(df)
    # zadanie 3
    # opisowe
    # zadanie 4
    # print(f"\nTrafność klasyfikacji z użyciem G-mean:")
    # results = evaluate_ensemble_methods(df)
    # print_results_zad4(results)
    # zadanie 5
    print(f"\nPorównanie wyników oryginalnych i znormalizowanych danych:")
    original_results, scaled_results = compare_original_vs_scaled(df)
    print_results_zad5(original_results, scaled_results)
    # zadanie 6
    # print(f"\nTuning VotingClassifier:")
    # best_score, best_params = tune_voting_classifier(df)
    # print(f"Najlepszy wynik G-mean: {best_score:.4f}")
    # print(f"Najlepsze parametry: {best_params}")
    # (mean_orig, std_orig), (mean_bal, std_bal) = evaluate_class_weight_effect(df, best_params)
    # print(f"\nOryginalny G-mean: {mean_orig:.4f} ± {std_orig:.4f}")
    # print(f"Z class_weight='balanced' G-mean: {mean_bal:.4f} ± {std_bal:.4f}")
    # print(f"Zysk z class_weight: {mean_bal - mean_orig:.4f}")
    # zadanie 7
    """
    print(f"\nŚrednia macierz pomyłek:")
    all_cm, mean_cm, std_cm = plot_avg_confusion_matrix(df)
    print("Średnia macierz pomyłek:", np.round(mean_cm, 2))
    print("Odchylenie standardowe:", np.round(std_cm, 2))
    accuracies = []
    for cm in all_cm:
        correct = np.trace(cm)  # suma elementów na przekątnej: TP + TN
        total = np.sum(cm)      # suma wszystkich elementów
        acc = correct / total
        accuracies.append(acc)

    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)
    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    """


