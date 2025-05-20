import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Klasyfikatory
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from imblearn.metrics import geometric_mean_score
from sklearn.base import is_classifier
from sklearn.base import clone
import inspect


def plot_value_distribution(df):
    plt.figure(figsize=(20, 10))
    sns.violinplot(data=df.iloc[:, :-1], inner="quart")
    plt.xticks(rotation=90)
    plt.title("Rozkład wartości atrybutów")
    plt.xlabel("Atrybuty")
    plt.ylabel("Wartości")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("rozklad-wartosci-atrybutow.png")
    
def plot_corelation_heat(df, zapisz_do=None):
    plt.figure(figsize=(16, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Macierz korelacji cech")
    plt.tight_layout()
    plt.savefig("korelacja.png")
    # plt.show()
    
def create_pca2d(df):
    pca_2d = PCA(n_components=2)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_scaled = StandardScaler().fit_transform(X)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    explained_variance_2d = pca_2d.explained_variance_ratio_.sum()
    
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=y, palette="coolwarm", alpha=0.7)
    
    handles, labels = scatter.get_legend_handles_labels()
    scatter.legend(handles=handles, labels=['Klasa 0', 'Klasa 1'], title='Klasy')
    
    plt.title(f'PCA 2D - Procent wariancji: {explained_variance_2d*100:.2f}%')
    plt.xlabel('Główna składowa 1')
    plt.ylabel('Główna składowa 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca2d.png")
    print(f'Procent wariancji zachowany przy rzutowaniu do 2D: {explained_variance_2d*100:.2f}%')
    
    
def create_pca3d(df):
    pca_3d = PCA(n_components=3)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_scaled = StandardScaler().fit_transform(X)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    explained_variance_3d = pca_3d.explained_variance_ratio_.sum()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_title(f'PCA 3D - Procent wariancji: {explained_variance_3d*100:.2f}%')
    ax.set_xlabel('Główna składowa 1')
    ax.set_ylabel('Główna składowa 2')
    ax.set_zlabel('Główna składowa 3')
    plt.tight_layout()
    plt.savefig("pca3d.png")
    print(f'Procent wariancji zachowany przy rzutowaniu do 3D: {explained_variance_3d*100:.2f}%')
    
    
def zad4(df):
    # G-mean scorer
    def gmean_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])
        return np.sqrt(sensitivity * specificity)

    gmean_scorer = make_scorer(gmean_score)

    # Wczytaj dane
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Lista klasyfikatorów
    classifiers = {
        'ZeroRule': DummyClassifier(strategy='most_frequent'),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(probability=True),
        'MLP': MLPClassifier(max_iter=1000),
        'GaussianNB': GaussianNB(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    # Przygotuj wyniki
    results = {clf_name: {'accuracy': [], 'gmean': [], 'roc_auc': []} for clf_name in classifiers}

    # 10-fold stratified CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Przetestuj klasyfikatory
    for name, clf in classifiers.items():
        print(f"Trening klasyfikatora: {name}")
        if name != 'ZeroRule':
            pipeline = ImbPipeline(steps=[('smote', SMOTE(random_state=42)), ('clf', clf)])
        else:
            pipeline = make_pipeline(clf)
        
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring={
                'accuracy': 'accuracy',
                'gmean': gmean_scorer,
                'roc_auc': 'roc_auc'
            },
            return_train_score=False
        )

        for metric in ['accuracy', 'gmean', 'roc_auc']:
            results[name][metric] = [np.mean(scores[f'test_{metric}']), np.std(scores[f'test_{metric}'])]

    # Konwersja wyników do DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy', 'gmean', 'roc_auc'])
    means = results_df.applymap(lambda x: x[0])
    stds = results_df.applymap(lambda x: x[1])

    # Wykres
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(classifiers))
    width = 0.25

    ax.bar(x - width, means['accuracy'], width, yerr=stds['accuracy'], label='Accuracy')
    ax.bar(x, means['gmean'], width, yerr=stds['gmean'], label='G-Mean')
    ax.bar(x + width, means['roc_auc'], width, yerr=stds['roc_auc'], label='ROC AUC')

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers.keys(), rotation=45)
    ax.set_title('Porównanie klasyfikatorów (SMOTE + CV)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('comparison.png')
    
def zad5_normalization_comparison(df):
    def gmean_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) else 0
        return np.sqrt(sensitivity * specificity)

    gmean_scorer = make_scorer(gmean_score)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    classifiers = {
        'ZeroRule': DummyClassifier(strategy='most_frequent'),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(probability=True),
        'MLP': MLPClassifier(max_iter=1000),
        'GaussianNB': GaussianNB(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    scalers = {
        'original': None,
        'standard': StandardScaler(),
        'power': PowerTransformer()
    }

    all_results = {}

    def run_experiment(scaler_key, scaler):
        results = {clf_name: {'accuracy': [], 'gmean': [], 'roc_auc': []} for clf_name in classifiers}
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for name, clf in classifiers.items():
            print(f"Trening ({scaler_key}) klasyfikatora: {name}")
            steps = []

            if scaler and name != 'ZeroRule':
                steps.append(('scaler', scaler))

            if name != 'ZeroRule':
                steps.append(('smote', SMOTE(random_state=42)))
                steps.append(('clf', clf))
                pipeline = ImbPipeline(steps=steps)
            else:
                pipeline = make_pipeline(clf)

            scores = cross_validate(
                pipeline, X, y,
                cv=cv,
                scoring={
                    'accuracy': 'accuracy',
                    'gmean': gmean_scorer,
                    'roc_auc': 'roc_auc'
                },
                return_train_score=False
            )

            for metric in ['accuracy', 'gmean', 'roc_auc']:
                results[name][metric] = [np.mean(scores[f'test_{metric}']), np.std(scores[f'test_{metric}'])]

        return results

    for key, scaler in scalers.items():
        all_results[key] = run_experiment(key, scaler)

    # Przekształcenie słownika wyników do DataFrame'ów
    def to_df(results_dict):
        return pd.DataFrame({clf: {m: results_dict[clf][m][0] for m in ['accuracy', 'gmean', 'roc_auc']} for clf in results_dict}).T

    df_orig = to_df(all_results['original'])
    df_std = to_df(all_results['standard'])
    df_pow = to_df(all_results['power'])

    # Różnice
    df_diff_std = df_std - df_orig
    df_diff_pow = df_pow - df_orig

    # Wykres różnic
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    classifiers_list = list(classifiers.keys())
    x = np.arange(len(classifiers_list))
    width = 0.25

    for i, (ax, df_diff, title) in enumerate(zip(
        axes,
        [df_diff_std, df_diff_pow],
        ['StandardScaler', 'PowerTransformer']
    )):
        ax.bar(x - width, df_diff['accuracy'], width, label='Accuracy Δ')
        ax.bar(x, df_diff['gmean'], width, label='G-Mean Δ')
        ax.bar(x + width, df_diff['roc_auc'], width, label='ROC AUC Δ')

        ax.set_xticks(x)
        ax.set_xticklabels(classifiers_list, rotation=45)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title(f'Różnice metryk ({title} - Oryginalne)')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig('normalization_comparison_extended.png')

    print("Wyniki oryginalnych danych:")
    print(df_orig)
    print("\nWyniki po standardyzacji:")
    print(df_std)
    print("\nWyniki po potęgowej transformacji:")
    print(df_pow)
    print("\nRóżnice po standardyzacji:")
    print(df_diff_std)
    print("\nRóżnice po potęgowej transformacji:")
    print(df_diff_pow)
    
def zad6(df):
    def gmean_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        return np.sqrt(sensitivity * specificity)

    gmean_scorer = make_scorer(gmean_score)

    def tune_rf(df):
        # Przygotuj dane
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Pipeline: PowerTransformer + SMOTE + RandomForest
        pipeline = ImbPipeline([
            ('scaler', PowerTransformer()),
            ('smote', SMOTE(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42))
        ])

        # Siatka hiperparametrów
        param_grid = {
            'rf__n_estimators': [50, 100, 150],
            'rf__max_depth': [5, 10, None],
            'rf__min_samples_split': [2, 5, 10]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring=gmean_scorer,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid.fit(X, y)
        best_model = grid.best_estimator_
        best_score = grid.best_score_
        best_params = grid.best_params_

        # Dane do heatmap (dla każdego max_depth osobno)
        results = pd.DataFrame(grid.cv_results_)

        for depth in param_grid['rf__max_depth']:
            print(f"Tworzenie heatmapy dla max_depth={depth}")
            depth_val = 'None' if depth is None else str(depth)

            if depth is None:
                subset = results[results['param_rf__max_depth'].isnull()]
            else:
                subset = results[results['param_rf__max_depth'] == depth]

            heatmap_data = subset.pivot(
                index='param_rf__n_estimators',
                columns='param_rf__min_samples_split',
                values='mean_test_score'
            )

            if heatmap_data.empty:
                print(f"⚠️ Brak danych do heatmapy dla max_depth={depth_val}, pominięto.")
                continue

            plt.figure(figsize=(8, 6))
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='YlGnBu')
            plt.title(f'G-Mean dla RandomForest (max_depth={depth_val})')
            plt.xlabel('min_samples_split')
            plt.ylabel('n_estimators')
            plt.tight_layout()
            plt.savefig(f'heatmap_rf_depth_{depth_val}.png')

        print(f"Najlepszy wynik G-Mean: {best_score}")
        print(f"Najlepsze hiperparametry: {best_params}")
        return best_score, best_params, results
    
    best_score, best_params, results = tune_rf(df)
    print(f"Najlepszy wynik: {best_score}")
    print(f"Najlepsze hiperparametry: {best_params}")
    print("Wyniki GridSearchCV:")
    print(results[['param_rf__n_estimators', 'param_rf__max_depth', 'param_rf__min_samples_split', 'mean_test_score']])
    
    pipeline_default = ImbPipeline([
        ('scaler', PowerTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))  # domyślne parametry
    ])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    score_default = cross_val_score(pipeline_default, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring=gmean_scorer).mean()
    print("G-Mean dla domyślnego RF:", score_default)
    print("Zysk z dostrojenia:", best_score - score_default)
    
def zad7(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]

    # Wykres
    plt.figure(figsize=(10, 6))
    plt.title("Ważność cech wg RandomForest")
    plt.bar(range(len(importances)), importances[sorted_idx], align="center")
    plt.xticks(range(len(importances)), feature_names[sorted_idx], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance_random_forest.png")

    # Lista cech z wagami
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[sorted_idx],
        'importance': importances[sorted_idx]
    })
    print(f"Random forest:\n{feature_importance_df}")
    
    # K-Best feature importance
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_

    sorted_idx = np.argsort(scores)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(scores)), scores[sorted_idx])
    plt.xticks(range(len(scores)), X.columns[sorted_idx], rotation=90)
    plt.title("K-Best dla cech")
    plt.tight_layout()
    plt.savefig("feature_importance_k_best.png")

    # Print K-Best feature importance
    k_best_df = pd.DataFrame({
        'feature': X.columns[sorted_idx],
        'score': scores[sorted_idx]
    })
    print(f"K-Best:\n{k_best_df}")
    
    # Permutation importance
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    r = permutation_importance(model, X, y, n_repeats=30, random_state=42, scoring='accuracy')

    sorted_idx = r.importances_mean.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.boxplot(r.importances[sorted_idx].T, vert=False, tick_labels=X.columns[sorted_idx])
    plt.title("Permutacyjna ważność cech")
    plt.tight_layout()
    plt.savefig("feature_importance_permutation.png")

    # Print Permutation importance
    perm_importance_df = pd.DataFrame({
        'feature': X.columns[sorted_idx],
        'mean_importance': r.importances_mean[sorted_idx],
        'std_importance': r.importances_std[sorted_idx]
    })
    print(f"Permutation importance:\n{perm_importance_df}")
    
def zad8(df):
    # Podział danych
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Definicje różnych wag
    weights = [
        {0: 1, 1: 1},              # Równe wagi
        'balanced',               # Wagi proporcjonalne do liczności klas
        {0: 1, 1: 5},              # Silne wsparcie dla klasy 1
        {0: 5, 1: 1},              # Silne wsparcie dla klasy 0
        {0: 0.5, 1: 2}             # Umiarkowana preferencja klasy 1
    ]

    weight_labels = ['equal', 'balanced', 'minority=5', 'majority=5', 'minority=2']

    fp_list = []
    fn_list = []

    for w, label in zip(weights, weight_labels):
        model = RandomForestClassifier(class_weight=w, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fp_list.append(fp)
        fn_list.append(fn)

        print(f"\nWagi: {label} ({w})")
        print("Macierz pomyłek:")
        print(cm)

    # Wykres FP i FN
    x = np.arange(len(weight_labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, fp_list, width, label='False Positives')
    plt.bar(x + width/2, fn_list, width, label='False Negatives')
    plt.xticks(x, weight_labels)
    plt.ylabel('Liczba błędów')
    plt.title('Wpływ class_weight w RandomForest na FP i FN')
    plt.legend()
    plt.tight_layout()
    plt.savefig("class_weight_fp_fn.png")
    
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.base import is_classifier, clone
import inspect

def zad9(df):
    def evaluate_classifiers(X, y, classifiers, class_weights=None, normalize=False):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        metrics = {
            'accuracy': [],
            'roc_auc': [],
            'g_mean': [],
        }

        best_accuracy = {'classifier': None, 'score': 0}
        best_roc_auc = {'classifier': None, 'score': 0}
        best_g_mean = {'classifier': None, 'score': 0}

        for clf_name, clf in classifiers.items():
            clf = clone(clf)  # nowy obiekt klasyfikatora

            # Ustawienie wag klas, jeśli obsługiwane
            if class_weights:
                try:
                    sig = inspect.signature(clf.__init__)
                    if 'class_weight' in sig.parameters:
                        clf.set_params(class_weight=class_weights)
                except (ValueError, TypeError):
                    pass

            accuracy_scores = []
            roc_auc_scores = []
            g_mean_scores = []
            confusion_matrices = []

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

                accuracy_scores.append(accuracy_score(y_test, y_pred))
                roc_auc_scores.append(roc_auc_score(y_test, y_prob) if y_prob is not None else 0)
                g_mean_scores.append(geometric_mean_score(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                confusion_matrices.append(cm)

            avg_accuracy = np.mean(accuracy_scores)
            avg_roc_auc = np.mean(roc_auc_scores)
            avg_g_mean = np.mean(g_mean_scores)

            std_accuracy = np.std(accuracy_scores)
            std_roc_auc = np.std(roc_auc_scores)
            std_g_mean = np.std(g_mean_scores)

            metrics['accuracy'].append(avg_accuracy)
            metrics['roc_auc'].append(avg_roc_auc)
            metrics['g_mean'].append(avg_g_mean)

            if avg_accuracy > best_accuracy['score']:
                best_accuracy = {'classifier': clf_name, 'score': avg_accuracy}
            if avg_roc_auc > best_roc_auc['score']:
                best_roc_auc = {'classifier': clf_name, 'score': avg_roc_auc}
            if avg_g_mean > best_g_mean['score']:
                best_g_mean = {'classifier': clf_name, 'score': avg_g_mean}

            print(f"\nKlasyfikator: {clf_name}")
            print(f"Średnia dokładność: {avg_accuracy:.4f} (std: {std_accuracy:.4f})")
            print(f"Średnia ROC AUC: {avg_roc_auc:.4f} (std: {std_roc_auc:.4f})")
            print(f"Średnia G-mean: {avg_g_mean:.4f} (std: {std_g_mean:.4f})")
            print("Średnia macierz pomyłek:")
            avg_cm = np.mean(confusion_matrices, axis=0).astype(int)
            print(avg_cm)
            print("-" * 40)

        print("\nNajlepszy klasyfikator według dokładności:")
        print(f"{best_accuracy['classifier']} - {best_accuracy['score']:.4f}")
        print("Najlepszy klasyfikator według ROC AUC:")
        print(f"{best_roc_auc['classifier']} - {best_roc_auc['score']:.4f}")
        print("Najlepszy klasyfikator według G-mean:")
        print(f"{best_g_mean['classifier']} - {best_g_mean['score']:.4f}")

        return metrics, best_accuracy, best_roc_auc, best_g_mean

    # Dane
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    # Klasyfikatory
    classifiers = {
        'ZeroRule': DummyClassifier(strategy='most_frequent'),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(probability=True),
        'MLP': MLPClassifier(max_iter=1000),
        'GaussianNB': GaussianNB(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    # Wagi klas
    class_weights = {
        'equal': {0: 1, 1: 1},
        'balanced': 'balanced',
        'minority=5': {0: 1, 1: 5},
        'majority=5': {0: 5, 1: 1},
        'minority=2': {0: 0.5, 1: 2},
    }

    # Ocena klasyfikatorów dla każdej z wag
    for weight_name, weight in class_weights.items():
        print(f"\n{'=' * 50}\nOcena klasyfikatorów z wagami: {weight_name}")
        evaluate_classifiers(X, y, classifiers, class_weights=weight, normalize=True)


if __name__ == "__main__":
    file_path = '151879-imbalanced.txt'
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)

    columns = [
        'dissim', 'el0', 'el1', 'el2', 'el3', 'el4', 'el5', 'el6', 'el7', 'el8', 'el9', 'el10', 'el11', 'el12', 'el13', 'el14',
        'el15', 'max_el', 'diff', 'diffplus', 'diffminus', 'std', 'diff/std**0.2', 'diff/std**0.6', 'diff/std**1',
        'diff/std**1.4', 'diff/std**2', 'maxmiddle', 'similarwindow_4.0', 'similarwindow_4.1', 'similarwindow_4.2',
        'similarwindow_4.3', 'stat0', 'stat1', 'stat2', 'stat3', 'stat4', 'stat5', 'stat6', 'stat7', 'stat8', 'stat9',
        'stat10', 'stat11', 'stat12', 'stat13', 'stat14', 'stat15', 'stat16', 'stat17', 'stat18', 'stat19', 'stat20',
        'stat21', 'stat22', 'stat23', 'stat24', 'stat25', 'stat26', 'stat27', 'stat28', 'stat29', 'stat30', 'stat31',
        'stat32', 'stat33', 'stat34', 'stat35', 'stat36', 'stat37', 'stat38', 'stat39', 'stat40', 'stat41', 'stat42',
        'stat43', 'stat44', 'class'
    ]

    df = pd.DataFrame(data, columns=columns)
    # plot_value_distribution(df)
    Xregr=data[:,0:-1]
    yregr=data[:,-1]
    
    # print("\nLiczba przykładów:", Xregr.shape[0])
    # print("Liczba atrybutów:", Xregr.shape[1])
    # print("Atrybuty klasy:", 1)

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # print(df.describe(include='all').T)
    
    # plot_corelation_heat(df)
    
    class_counts = df['class'].value_counts()
    # print(f"Rozkład klas:", class_counts)
    
    # create_pca2d(df)
    # create_pca3d(df)
    
    # zad4(df)
    
    # zad5_normalization_comparison(df)
    
    # zad6(df)
    
    # zad7(df)
    
    # zad8(df)
    
    zad9(df)
    
    