import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# ── Dynamic Absolute Path Fixing ──────────────────────────────────────────
# This finds your root directory ('clocked-ai') automatically
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(BASE, "models")
os.makedirs(models_dir, exist_ok=True)

# FIXED: Safely builds absolute path targeting clocked-ai/data/features.csv
features_path = os.path.join(BASE, "data", "features.csv")
if not os.path.exists(features_path):
    # Fallback check in case features.csv is inside a subfolder nested at clocked-ai/data/features/features.csv
    features_path = os.path.join(BASE, "data", "features", "features.csv")

df = pd.read_csv(features_path)

# ── Feature Selection ─────────────────────────────────────────────────────
feature_cols = [col for col in df.columns if col != 'model']
X = df[feature_cols]
target_col = 'model_name' if 'model_name' in df.columns else 'model'
y = df[target_col]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("=" * 50)
print("  DATA DISTRIBUTION")
print("=" * 50)
print(f"Training set size : {X_train.shape} rows")
print(f"Testing set size  : {X_test.shape} rows")
print(f"Target LLM Classes: {list(le.classes_)}\n")

# ── Define Core Classifiers ───────────────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=2, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='mlogloss', random_state=42)
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))

base_classifiers = [
    ("rf", rf_model),
    ("xgb", xgb_model),
    ("svm", svm_pipeline)
]

stacking_meta_model = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, eval_metric='mlogloss', random_state=42),
    cv=3,
    n_jobs=-1
)

all_classifiers = {
    "Random Forest": rf_model,
    "XGBoost":       xgb_model,
    "SVM (Scaled)":  svm_pipeline,
    "Stacking Ensemble": stacking_meta_model
}

print("=" * 50)
print("  MODEL TRAINING & LEAK-FREE VALIDATION")
print("=" * 50)

results = {}

for name, clf in all_classifiers.items():
    print(f"Evaluating {name}...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model':    clf,
        'accuracy': acc,
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'y_pred':   y_pred
    }
    print(f"  CV Accuracy (3-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Holdout Test Accuracy: {acc:.3f} ({acc*100:.1f}%)\n")

# ── Select Best Model ─────────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['cv_mean'])
best = results[best_name]

print("=" * 50)
print(f"WINNING MODEL: {best_name}")
print(f"  Optimized CV Score : {best['cv_mean']:.3f} +/- {best['cv_std']:.3f}")
print(f"  Holdout Test Score : {best['accuracy']:.3f}")
print("=" * 50)

# ── Detailed Report ───────────────────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'], target_names=le.classes_))

# ── Confusion Matrix (% Normalised) ───────────────────────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
cm_sum = cm.sum(axis=1, keepdims=True)
cm_pct = np.divide(cm.astype(float), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100

label_mapping = {
    'gemma-4-26b-a4b-it': 'Gemma 4 26B',
    'llama-3.3-70b-versatile': 'Llama 70B',
    'openai/gpt-oss-120b': 'GPT 120B',
    'qwen/qwen3-32b': 'Qwen 32B'
}

display_labels = [label_mapping.get(cls, cls) for cls in le.classes_]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_pct, 
    annot=True, 
    fmt='.1f', 
    cmap='Blues',
    xticklabels=display_labels,  
    yticklabels=display_labels    
)
plt.title(f'Normalized Confusion Matrix (%)\n{best_name}', fontsize=12)
plt.ylabel('Actual Model Identity')
plt.xlabel('Predicted Model Identity')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "confusion_matrix.png"), dpi=150)
plt.close()
print("Confusion matrix saved with custom labels to models/confusion_matrix.png")


# ── Advanced Feature Importance Extraction ────────────────────────────────
actual_model = best['model']
if best_name == "Stacking Ensemble":
    actual_model = stacking_meta_model.named_estimators_['rf']

if hasattr(actual_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': actual_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Critical Stylometric Fingerprint Features:")
    print(importance_df.head(10).to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x='importance', y='feature', palette='viridis')
    plt.title(f'Top 10 Features for LLM Fingerprinting ({best_name})', fontsize=12)
    plt.xlabel('Normalized Importance Weights')
    plt.ylabel('Feature Columns')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print("Feature importance plot saved to models/feature_importance.png")
else:
    print(f"\nNote: Selected winner ({best_name}) doesn't natively expose feature weights.")

# ── Safe Multi-Object Model Pickling ───────────────────────────────────────
with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
    pickle.dump({
        'model': best['model'], 
        'encoder': le,
        'features': feature_cols, 
        'name': best_name
    }, f)
print(f"\nSaved secure payload array to models/best_model.pkl")
print("All done! Execution complete.")
