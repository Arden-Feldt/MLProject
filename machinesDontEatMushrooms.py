# mushroom_classifier.py
# Authors: Ash Arun, Arden Feldt, Mo Murea

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Dataset ===
data = pd.read_csv("mushrooms.csv")

# === Encode Categorical Features ===
# All columns are categorical, so we use LabelEncoder
label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# === Split Features and Target ===
X = data.drop('class', axis=1)
y = data['class']  # 0 = edible, 1 = poisonous (depending on encoding)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Model 1: Decision Tree ===
tree = DecisionTreeClassifier(
    criterion='entropy', max_depth=None, min_samples_leaf=2, random_state=42
)
tree.fit(X_train, y_train)

# === Model 2: Logistic Regression ===
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# === Cross-Validation (K-Fold) ===
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_tree_scores = cross_val_score(tree, X, y, cv=kfold, scoring='roc_auc')
cv_log_scores = cross_val_score(log_reg, X, y, cv=kfold, scoring='roc_auc')

print("=== K-Fold AUC Scores ===")
print(f"Decision Tree: {cv_tree_scores.mean():.4f} ± {cv_tree_scores.std():.4f}")
print(f"Logistic Regression: {cv_log_scores.mean():.4f} ± {cv_log_scores.std():.4f}")

# === Predictions ===
y_pred_tree = tree.predict(X_test)
y_pred_log = log_reg.predict(X_test)

# === Evaluation Metrics ===
print("\n=== Classification Report: Decision Tree ===")
print(classification_report(y_test, y_pred_tree))

print("\n=== Classification Report: Logistic Regression ===")
print(classification_report(y_test, y_pred_log))

# === Confusion Matrices ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Decision Tree Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Logistic Regression Confusion Matrix')
plt.tight_layout()
plt.show()

# === ROC Curve & AUC ===
y_prob_tree = tree.predict_proba(X_test)[:, 1]
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
roc_auc_tree = auc(fpr_tree, tpr_tree)
roc_auc_log = auc(fpr_log, tpr_log)

plt.figure(figsize=(7, 5))
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {roc_auc_tree:.3f})")
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_log:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# === Optional: Visualize Decision Tree ===
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=["Edible", "Poisonous"], fontsize=8)
plt.title("Decision Tree Visualization")
plt.show()
