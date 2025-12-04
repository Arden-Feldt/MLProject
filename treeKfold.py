
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

k = 10  # num  folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

acc_scores, prec_scores, rec_scores, f1_scores, auc_scores = [], [], [], [], []
mean_fpr = np.linspace(0, 1, 100)
tprs = []

fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=2,
        random_state=42
    )
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    y_prob = tree.predict_proba(X_test)[:, 1]

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

print("\n=== Cross-Validation Summary ===")
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
plt.plot(mean_fpr, mean_tpr, color='blue', label=f"Mean ROC (AUC = {mean_auc:.3f})", lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Decision Tree K-Fold ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

print("\nInterpretation:")
print("If the cross-validation AUC and accuracy are both near 1.0 with low variance,")
print("the model may still be overfitting. Try limiting tree depth (max_depth) or increasing min_samples_leaf.")
print("If accuracy drops slightly but variance narrows, the model is likely generalizing better.")
