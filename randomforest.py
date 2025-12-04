import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)

data = pd.read_csv("mushrooms.csv")

label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(['class', 'odor', 'spore-print-color', 'gill-color'], axis=1)
y = data['class']

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

acc_scores, prec_scores, rec_scores, f1_scores, auc_scores = [], [], [], [], []
mean_fpr = np.linspace(0, 1, 100)
tprs = []

fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc_scores.append(accuracy_score(y_test, y_pred))
    prec_scores.append(precision_score(y_test, y_pred))
    rec_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)

    print(f"Fold {fold} — AUC: {roc_auc:.3f}")
    fold += 1

print("\n=== Random Forest Cross-Validation Summary ===")
print(f"Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
print(f"Recall:    {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
print(f"F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"AUC:       {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

plt.figure(figsize=(8, 6))
for i, tpr in enumerate(tprs):
    plt.plot(mean_fpr, tpr, alpha=0.2, label=f"Fold {i+1}")
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='darkorange', label=f"Mean ROC (AUC = {mean_auc:.3f})", lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Random Forest K-Fold ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

rf_final = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X, y)

importances = rf_final.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(y='Feature', x='Importance', data=feature_importances.head(15), palette='viridis')
plt.title("Top 15 Most Important Features — Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n=== Top 10 Important Features ===")
print(feature_importances.head(10))

top_features = feature_importances.head(5)['Feature'].tolist()
print("\nInterpretation:")
print(f"Random Forest identified {', '.join(top_features)} as the most predictive features.")
print("These features dominate the model’s decision-making process, often reflecting strong biological signals.")
print("If 'odor' appears near the top, that aligns with known biological separability in the dataset.")
print("Removing it or adding noise can help test the model’s robustness and simulate real-world uncertainty.")
