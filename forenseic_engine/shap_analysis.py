import os
import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load payload data and drop structural columns ───────────────
with open(os.path.join(BASE, "models", "best_model.pkl"), "rb") as f:
    payload = pickle.load(f)

# FIXED: Extract the actual underlying model object and saved feature columns
model = payload['model']
feature_cols = payload['features']

with open(os.path.join(BASE, "models", "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

df = pd.read_csv(os.path.join(BASE, "data", "features.csv") if os.path.exists(os.path.join(BASE, "data", "features.csv")) else os.path.join(BASE, "data", "features", "features.csv"))

# FIXED: Filter X to exactly match the 15 features used during training
X = df[feature_cols]

models_dir = os.path.join(BASE, "models")
shap_dir = os.path.join(models_dir, "shap")
os.makedirs(shap_dir, exist_ok=True)

# ── SHAP Explainer ─────────────────────────────────────────────
print("Running SHAP analysis...")

X_sample = shap.sample(X, 50, random_state=42)

def predict_fn(x):
    x_df = pd.DataFrame(x, columns=feature_cols)
    return model.predict_proba(x_df)

# Background dataset (used as SHAP reference)
background = shap.sample(X, 50, random_state=42)

explainer = shap.KernelExplainer(
    predict_fn,
    background
)

# Explain ALL 199 samples
shap_values = explainer.shap_values(X)


# ── Summary plot (beeswarm) ────────────────────────────────────
print("Saving beeswarm summary plot...")
plt.figure(figsize=(10, 7))
# Multiclass summary plots require passing the list array directly
shap.summary_plot(
    shap_values,
    X,
    class_names=le.classes_,
    show=False
)
plt.title("Multi-class SHAP Summary Impact Matrix", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Bar plot (mean absolute SHAP) ─────────────────────────────
print("Saving bar plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X,
    class_names=le.classes_,
    plot_type="bar",
    show=False
)
plt.title("Mean Absolute SHAP Feature Weightings Across Classes", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Per-class SHAP plots ───────────────────────────────────────
print("Saving per-class plots...")
for i, class_name in enumerate(le.classes_):
    plt.figure(figsize=(10, 6))
    
    # FIXED: Extract individual class vectors cleanly using slicing notation [:, :, i]
    # to maintain full Explainer object metadata across modern SHAP structures
    class_shap_values = shap_values[:, :, i]
    
    shap.summary_plot(
        class_shap_values,
        X,
        show=False,
        plot_type="dot"
    )
    plt.title(f"Stylometric Feature Impact Profile — {class_name}", fontsize=13, pad=15)
    plt.tight_layout()
    safe_name = class_name.replace("/", "_").replace(".", "_")
    plt.savefig(os.path.join(shap_dir, f"shap_{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

print(f"\nAll SHAP plots saved successfully to: models/shap/")
print("Done!")
