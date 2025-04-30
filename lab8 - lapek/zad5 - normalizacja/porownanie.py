import matplotlib.pyplot as plt
import numpy as np

# Nazwy klasyfikatorów
classifiers = ['KNeighbors', 'DecisionTree', 'RandomForest', 'SVC', 'MLP', 'GaussianNB', 'QDA', 'ZeroRule']

# Średnie i odchylenia standardowe: oryginalne dane
mean_with_smote = {
    'accuracy': [0.9715, 0.9809, 0.9897, 0.98, 0.9881, 0.9591, 0.9889, 0.0301],
    'gmean': [0.9493, 0.8791, 0.9450, 0.9570, 0.9235, 0.9445, 0.9444, 0.0],
    'roc_auc': [0.9498, 0.8854, 0.9463, 0.9574, 0.9262, 0.9450, 0.9459, 0.5],
}
std_with_smote = {
    'accuracy': [0.0047, 0.0045, 0.0027, 0.0039, 0.0035, 0.0033, 0.0017, 0.0003],
    'gmean': [0.0219, 0.0287, 0.0211, 0.0169, 0.0241, 0.0264, 0.0269, 0.0],
    'roc_auc': [0.0212, 0.0256, 0.0197, 0.0164, 0.0221, 0.0259, 0.0255, 0.0],
}
mean_without_smote = {
    'accuracy': [0.9915, 0.9851, 0.9904, 0.9779, 0.9908, 0.9588, 0.987, 0.9699],
    'gmean': [0.9088, 0.8511, 0.8893, 0.9625, 0.9122, 0.9412, 0.9519, 0.0],
    'roc_auc': [0.9136, 0.8619, 0.8968, 0.9628, 0.9163, 0.9417, 0.9530, 0.5],
}
std_without_smote = {
    'accuracy': [0.0032, 0.0042, 0.0034, 0.0044, 0.0029, 0.0044, 0.0014, 0.0003],
    'gmean': [0.0448, 0.0365, 0.0589, 0.0153, 0.0362, 0.0235, 0.0243, 0.0],
    'roc_auc': [0.0402, 0.0302, 0.0508, 0.0150, 0.0320, 0.0229, 0.0231, 0.0],
}

# Funkcja do rysowania wykresów
def plot_with_error_bars(metric_name, ylabel):
    x = np.arange(len(classifiers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, mean_with_smote[metric_name], width, yerr=std_with_smote[metric_name], label='Z SMOTE', capsize=5)
    ax.bar(x + width/2, mean_without_smote[metric_name], width, yerr=std_without_smote[metric_name], label='Bez SMOTE', capsize=5)

    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} dla różnych klasyfikatorów (z i bez SMOTE)')
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

plot_with_error_bars('accuracy', 'Accuracy')
plot_with_error_bars('gmean', 'G-mean')
plot_with_error_bars('roc_auc', 'ROC AUC')
