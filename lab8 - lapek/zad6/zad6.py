import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer

# Wczytanie danych
df = pd.read_csv("151879-imbalanced.txt", sep="\t")
y = df['class']
X = df.drop(columns='class')

# Metryka do optymalizacji
gmean_scorer = make_scorer(geometric_mean_score)

# Klasyfikator i pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Zakres hiperparametrów
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_split': [2, 5],
    'clf__max_features': ['sqrt', 'log2']
}

# GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=gmean_scorer, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Wyniki
results = pd.DataFrame(grid_search.cv_results_)

# Zapis wyników
results.to_csv("rf_gridsearch_results.csv", index=False)


# Heatmapy – np. dla n_estimators vs max_depth przy różnych max_features
def plot_heatmap(data, param_x, param_y, fixed_param, fixed_value, metric='mean_test_score'):
    subset = data[data[f'param_{fixed_param}'] == fixed_value]
    pivot = subset.pivot_table(
        index=f'param_{param_y}',
        columns=f'param_{param_x}',
        values=metric
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f'{metric} for {fixed_param}={fixed_value}')
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.tight_layout()
    plt.savefig(f"heatmap_{param_x}_vs_{param_y}_with_{fixed_param}_{fixed_value}.png")
    plt.close()


# Przykład – tworzenie heatmap dla różnych wartości max_features
for max_feat in ['sqrt', 'log2']:
    plot_heatmap(
        results,
        param_x='clf__n_estimators',
        param_y='clf__max_depth',
        fixed_param='clf__max_features',
        fixed_value=max_feat
    )
