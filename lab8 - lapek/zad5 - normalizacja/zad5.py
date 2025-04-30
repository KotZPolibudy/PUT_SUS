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

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Wczytanie danych
df = pd.read_csv("151879-imbalanced.txt", sep="\t")
y = df['class']
X = df.drop(columns='class')

# Definicja klasyfikatorów
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

scorers = {
    'accuracy': make_scorer(accuracy_score),
    'gmean': make_scorer(geometric_mean_score),
    'roc_auc': make_scorer(roc_auc_score, multi_class='ovo')
}

# Walidacja krzyżowa
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)


def evaluate_classifiers(apply_smote):
    results = {}
    for name, clf in classifiers.items():
        print(name, apply_smote)
        if apply_smote:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('clf', clf)
            ])
        else:
            pipeline = Pipeline([  # klasyczny pipeline bez SMOTE
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=False)
        results[name] = scores
    return results


# Obliczenia
results_without_smote = evaluate_classifiers(apply_smote=False)
print("progress!")
results_with_smote = evaluate_classifiers(apply_smote=True)


# Agregacja wyników
def aggregate_results(results):
    mean_scores, std_scores = {}, {}
    for name, scores in results.items():
        mean_scores[name] = {metric: np.mean(scores[f'test_{metric}']) for metric in scorers}
        std_scores[name] = {metric: np.std(scores[f'test_{metric}']) for metric in scorers}
    return mean_scores, std_scores


mean_with_smote, std_with_smote = aggregate_results(results_with_smote)
mean_without_smote, std_without_smote = aggregate_results(results_without_smote)

print(mean_with_smote, std_with_smote, mean_without_smote, std_without_smote)


# Zapis wyników do pliku txt
def save_results_to_txt(mean_with, std_with, mean_without, std_without, filename="results.txt"):
    with open(filename, "w") as f:
        for clf_name in mean_with:
            f.write(f"Classifier: {clf_name}\n")
            for metric in scorers:
                mw = mean_with[clf_name][metric]
                sw = std_with[clf_name][metric]
                mn = mean_without[clf_name][metric]
                sn = std_without[clf_name][metric]
                f.write(f"  {metric}:\n")
                f.write(f"    With SMOTE   : mean={mw:.4f}, std={sw:.4f}\n")
                f.write(f"    Without SMOTE: mean={mn:.4f}, std={sn:.4f}\n")
            f.write("\n")


save_results_to_txt(mean_with_smote, std_with_smote, mean_without_smote, std_without_smote)


# Generowanie i zapisywanie wykresów
def plot_metric(metric, mean_with, mean_without, filename):
    labels = list(mean_with.keys())
    with_values = [mean_with[clf][metric] for clf in labels]
    without_values = [mean_without[clf][metric] for clf in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, without_values, width, label='Without SMOTE')
    bars2 = ax.bar(x + width / 2, with_values, width, label='With SMOTE')

    ax.set_ylabel(metric)
    ax.set_title(f'{metric.upper()} comparison with/without SMOTE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_combined_results(mean_with, std_with, mean_without, std_without, filename="combined_metrics.png"):
    metrics = list(scorers.keys())
    classifiers = list(mean_with.keys())
    x = np.arange(len(classifiers))

    width = 0.12  # szerokość pojedynczego słupka
    offsets = {
        ('accuracy', 'without'): -2 * width,
        ('accuracy', 'with'): -1 * width,
        ('gmean', 'without'): 0,
        ('gmean', 'with'): width,
        ('roc_auc', 'without'): 2 * width,
        ('roc_auc', 'with'): 3 * width
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for metric in metrics:
        for mode in ['without', 'with']:
            means = [mean_with[clf][metric] if mode == 'with' else mean_without[clf][metric] for clf in classifiers]
            stds = [std_with[clf][metric] if mode == 'with' else std_without[clf][metric] for clf in classifiers]
            label = f"{metric} ({'SMOTE' if mode == 'with' else 'No SMOTE'})"
            pos = x + offsets[(metric, mode)]
            ax.bar(pos, means, width=width, yerr=stds, capsize=4, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Porównanie klasyfikatorów (3 metryki, z/bez SMOTE, z odchyleniami)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Tworzenie wykresów dla każdej metryki
for metric in scorers:
    plot_metric(
        metric,
        mean_with_smote,
        mean_without_smote,
        filename=f"{metric}_comparison.png"
    )

plot_combined_results(mean_with_smote, std_with_smote, mean_without_smote, std_without_smote)
