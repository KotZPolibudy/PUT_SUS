import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline

# Wczytanie danych
df = pd.read_csv("151879-imbalanced.txt", sep="\t")
y = df['class']
X = df.drop(columns='class')

# Klasyfikatory
classifiers = {
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
    'RandomForest': RandomForestClassifier(class_weight='balanced'),
    'SVC': SVC(class_weight='balanced', probability=True),
    'MLP': MLPClassifier(max_iter=500),
    'GaussianNB': GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'ZeroRule': DummyClassifier(strategy='most_frequent')
}

# Metryki
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'gmean': make_scorer(geometric_mean_score),
    'roc_auc': make_scorer(roc_auc_score, multi_class='ovo')
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

def evaluate_classifiers(apply_smote, scaler):
    results = {}
    for name, clf in classifiers.items():
        steps = []
        if scaler is not None:
            steps.append(('scaler', scaler))
        if apply_smote:
            steps.append(('smote', SMOTE(random_state=42)))
        steps.append(('clf', clf))

        pipeline = ImbPipeline(steps)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=False)
        results[name] = scores
    return results

# Eksperymenty
scalers = {
    'original': None,
    'standard': StandardScaler(),
    'power': PowerTransformer()
}

results = {}
for scaler_name, scaler in scalers.items():
    results[scaler_name] = {
        'with_smote': evaluate_classifiers(apply_smote=True, scaler=scaler),
        'without_smote': evaluate_classifiers(apply_smote=False, scaler=scaler)
    }

# Agregacja
def aggregate_results(results):
    mean_scores, std_scores = {}, {}
    for name, scores in results.items():
        mean_scores[name] = {metric: np.mean(scores[f'test_{metric}']) for metric in scorers}
        std_scores[name] = {metric: np.std(scores[f'test_{metric}']) for metric in scorers}
    return mean_scores, std_scores

aggregated = {}
for scaler_name in scalers:
    aggregated[scaler_name] = {
        'with': aggregate_results(results[scaler_name]['with_smote']),
        'without': aggregate_results(results[scaler_name]['without_smote'])
    }

# Wykres różnic między normalizacjami
def plot_differences(metric, base_mean, comp_mean, title, filename):
    labels = list(base_mean.keys())
    diffs = [comp_mean[clf][metric] - base_mean[clf][metric] for clf in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, diffs, color='skyblue')

    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f"Difference in {metric}")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Różnice: standard vs original, power vs original
for mode in ['with', 'without']:
    for metric in scorers:
        plot_differences(
            metric,
            aggregated['original'][mode][0],
            aggregated['standard'][mode][0],
            title=f"Znormalizowane (StandardScaler) vs Oryginalne — {metric} ({mode})",
            filename=f"diff_standard_vs_original_{metric}_{mode}.png"
        )
        plot_differences(
            metric,
            aggregated['original'][mode][0],
            aggregated['power'][mode][0],
            title=f"Znormalizowane (PowerTransformer) vs Oryginalne — {metric} ({mode})",
            filename=f"diff_power_vs_original_{metric}_{mode}.png"
        )
