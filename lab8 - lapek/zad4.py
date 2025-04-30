import pandas as pd
import numpy as np

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

import matplotlib.pyplot as plt

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
    'QDA': QuadraticDiscriminantAnalysis()
}

# Skorery
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'gmean': make_scorer(geometric_mean_score),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo')
}

# Walidacja krzyżowa
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

# Wyniki
results = {}

for name, clf in classifiers.items():
    # Dla każdego klasyfikatora użyj SMOTE, by przeciwdziałać niezrównoważeniu
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', clf)
    ])
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=False)
    results[name] = scores

# Klasyfikator ZeroRule (zawsze przewiduje najczęstszą klasę)
zero_rule = DummyClassifier(strategy='most_frequent')
zr_scores = cross_validate(zero_rule, X, y, cv=cv, scoring=scorers)
results['ZeroRule'] = zr_scores

# Agregacja metryk
mean_scores = {}
std_scores = {}
for name, scores in results.items():
    mean_scores[name] = {metric: np.mean(scores[f'test_{metric}']) for metric in scorers}
    std_scores[name] = {metric: np.std(scores[f'test_{metric}']) for metric in scorers}

(mean_scores, std_scores)
