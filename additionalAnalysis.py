import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)

data = pd.read_csv("mushrooms.csv")

label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop('class', axis=1)
y = data['class'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    auc_scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
    cv_results[name] = auc_scores
    print(f"{name}: AUC = {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr
    }

for name, res in results.items():
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, res["y_pred"]))

plt.figure(figsize=(8, 6))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

feature_importances = pd.DataFrame(index=X.columns)

for name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
    model = results[name]["model"]
    importances = model.feature_importances_
    feature_importances[name] = importances

feature_importances["Mean Importance"] = feature_importances.mean(axis=1)
feature_importances.sort_values("Mean Importance", ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(
    y=feature_importances.index[:15],
    x=feature_importances["Mean Importance"][:15],
    palette="viridis"
)
plt.title("Top 15 Most Important Features Across Tree-Based Models")
plt.xlabel("Mean Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n=== Most Important Features ===")
print(feature_importances.head(10))

top_features = feature_importances.head(5).index.tolist()
print("\n=== Analysis ===")
print(f"Across all tree-based models, the most influential features were: {', '.join(top_features)}.")
print("Features like 'odor' and 'spore-print-color' often dominate importance rankings,")
print("which aligns with biological knowledge—odor and spore traits are key in distinguishing toxic from edible mushrooms.")
print("Visual traits like 'cap-color' or 'gill-color' were moderately predictive,")
print("supporting the idea that simpler, observable features can be useful in clinical triage.")
