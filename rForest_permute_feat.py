import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("mushrooms.csv")

label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(['class', 'odor', 'spore-print-color', 'gill-color'], axis=1)
y = data['class']

def evaluate_model(X, y, title="Normal Data"):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
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

print("\n=== Feature Permutation Robustness Test ===")
baseline_acc = acc_real
feature_drops = []

for feature in X.columns:
    X_permuted = X.copy()
    X_permuted[feature] = np.random.permutation(X_permuted[feature])

    acc_shuffled, _ = evaluate_model(X_permuted, y, f"Permuted Feature: {feature}")
    drop = baseline_acc - acc_shuffled
    feature_drops.append((feature, drop))

drop_df = pd.DataFrame(feature_drops, columns=["Feature", "Accuracy Drop"])
drop_df = drop_df.sort_values("Accuracy Drop", ascending=False)

print("\n=== Comparison Summary ===")
print(f"Original Accuracy: {acc_real:.4f}")
print(f"Permuted Labels Accuracy: {acc_perm:.4f}")
print("\nTop 10 Most Critical Features (by Accuracy Drop):")
print(drop_df.head(10))

plt.figure(figsize=(6, 4))
plt.bar(["Normal Data", "Permuted Labels"], [acc_real, acc_perm], color=['green', 'red'])
plt.ylabel("Accuracy")
plt.title("Model Robustness — Normal vs. Randomized Labels")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(y="Feature", x="Accuracy Drop", data=drop_df.head(15), palette="mako")
plt.title("Feature Importance via Permutation (Accuracy Drop)")
plt.xlabel("Decrease in Accuracy when Feature is Shuffled")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("1. If the permuted-label accuracy ≈ 0.5, the model is learning genuine structure.")
print("2. Features with the largest accuracy drops are most crucial to correct predictions.")
print("3. For example, if 'stalk-root' or 'cap-color' show large drops, they strongly influence edibility classification.")
print("4. This test verifies both biological interpretability and model robustness.")
