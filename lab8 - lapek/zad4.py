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
    'MLP': MLPClassifier(max_iter=1000),
    'GaussianNB': GaussianNB(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'ZeroRule': DummyClassifier(strategy='most_frequent')
}

# Skorery
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'gmean': make_scorer(geometric_mean_score),
    'roc_auc': make_scorer(roc_auc_score, multi_class='ovo')
}

# Walidacja krzyżowa
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

# Funkcja testująca klasyfikatory
def evaluate_classifiers(apply_smote):
    results = {}
    for name, clf in classifiers.items():
        print(name, apply_smote)
        if apply_smote:
            pipeline = ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', clf)])
        else:
            pipeline = ImbPipeline([('clf', clf)])
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=False)
        results[name] = scores
    return results

# Obliczenia
results_with_smote = evaluate_classifiers(apply_smote=True)
print("progress!")
results_without_smote = evaluate_classifiers(apply_smote=False)

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
