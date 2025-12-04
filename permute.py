import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

data = pd.read_csv("mushrooms.csv")

label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(['class', 'odor', 'spore-print-color', 'gill-color'], axis=1)
y = data['class']

def evaluate_model(X, y, title="Normal Data"):
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    acc_scores, auc_scores = [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))

    print(f"\n=== {title} ===")
    print(f"Mean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Mean AUC:      {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    return np.mean(acc_scores), np.mean(auc_scores)

acc_real, auc_real = evaluate_model(X, y, "Normal Mushroom Data")

y_permuted = pd.Series(np.random.permutation(y), index=y.index)
acc_perm, auc_perm = evaluate_model(X, y_permuted, "Permuted Labels (Randomized Targets)")


print("\n=== Comparison Summary ===")
print(f"Original Accuracy: {acc_real:.4f}")
print(f"Permuted Accuracy: {acc_perm:.4f}")
print(f"Original AUC:      {auc_real:.4f}")
print(f"Permuted AUC:      {auc_perm:.4f}")

plt.figure(figsize=(6, 4))
plt.bar(["Normal Data", "Permuted Labels"], [acc_real, acc_perm], color=['green', 'red'])
plt.ylabel("Accuracy")
plt.title("Model Robustness Check — Random Forest")
plt.ylim(0, 1)
plt.show()

print("\nInterpretation:")
print("If the permuted-label model performs near 0.5 AUC and ~50% accuracy,")
print("the Random Forest is learning real structure from the data.")
print("If accuracy remains high after permutation, the model is likely overfitting or data leakage exists.")
