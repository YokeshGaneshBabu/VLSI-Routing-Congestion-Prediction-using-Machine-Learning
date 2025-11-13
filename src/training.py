import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("🚀 ENHANCED CONGESTION PREDICTION - TRAINING PIPELINE")
print("="*70)

# ============================================================
# STEP 1: LOAD THE ENHANCED DATASET
# ============================================================
print("\n📂 Loading enhanced dataset...")

# Assuming you've already generated the dataset using the generator
# If not, uncomment and run the generator first:
"""
from realistic_congestion_generator import RealisticCongestionGenerator

generator = RealisticCongestionGenerator(grid_size=64, num_layouts=10)
df, congestion_maps = generator.generate_large_dataset(
    tech_nodes=['7nm', '14nm', '28nm']
)
df.to_csv('realistic_congestion_dataset.csv', index=False)
"""

# Load dataset
try:
    df = pd.read_csv('realistic_congestion_dataset.csv')
    print(f"✅ Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("❌ Dataset not found! Please run the generator first.")
    print("   Run the RealisticCongestionGenerator code to create the dataset.")
    exit()

# ============================================================
# STEP 2: DATA EXPLORATION
# ============================================================
print("\n" + "="*70)
print("📊 DATA EXPLORATION")
print("="*70)

print(f"\n📋 Dataset Info:")
print(f"   Total samples: {len(df):,}")
print(f"   Features: {len(df.columns) - 3}")  # Exclude target, tech_node, layout_id
print(f"   Layouts: {df['layout_id'].nunique()}")
print(f"   Technology nodes: {df['tech_node'].unique()}")

print(f"\n📈 Congestion Statistics:")
print(df['congestion'].describe())

print(f"\n🔢 Feature Summary:")
print(df.drop(['layout_id', 'tech_node', 'congestion'], axis=1).describe())

# Visualize congestion distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution
axes[0].hist(df['congestion'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Congestion Value', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Congestion Distribution', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# By tech node
for tech in df['tech_node'].unique():
    subset = df[df['tech_node'] == tech]['congestion']
    axes[1].hist(subset, bins=30, alpha=0.5, label=tech, edgecolor='black')
axes[1].set_xlabel('Congestion Value', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Congestion by Tech Node', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Boxplot by tech node
tech_data = [df[df['tech_node'] == tech]['congestion'].values for tech in df['tech_node'].unique()]
axes[2].boxplot(tech_data, labels=df['tech_node'].unique())
axes[2].set_xlabel('Technology Node', fontsize=11)
axes[2].set_ylabel('Congestion Value', fontsize=11)
axes[2].set_title('Congestion Range by Tech', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('congestion_distribution.png', dpi=150, bbox_inches='tight')
print("\n✅ Distribution plot saved as 'congestion_distribution.png'")
plt.show()

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*70)
print("🔧 FEATURE ENGINEERING")
print("="*70)

# Encode technology node
le = LabelEncoder()
df['tech_node_encoded'] = le.fit_transform(df['tech_node'])
print(f"✅ Technology nodes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Create classification labels
def classify_congestion(c):
    if c < 0.8:
        return "Green"    # Safe
    elif c < 1.5:
        return "Yellow"   # Warning
    else:
        return "Red"      # Critical

df['congestion_class'] = df['congestion'].apply(classify_congestion)

print(f"\n📊 Class Distribution:")
print(df['congestion_class'].value_counts())
print(f"\n   Class percentages:")
print(df['congestion_class'].value_counts(normalize=True) * 100)

# ============================================================
# STEP 4: PREPARE DATA FOR ML
# ============================================================
print("\n" + "="*70)
print("📦 PREPARING TRAIN/TEST SPLIT")
print("="*70)

# Select features (exclude metadata and target)
feature_cols = [
    'density', 'pin_density', 'fanout', 'macro', 'capacity', 'macro_proximity',
    'wire_length', 'clock_region', 'drc_violations', 'net_criticality',
    'power_density', 'temperature', 'tech_node_encoded'
]

X = df[feature_cols]
y_reg = df['congestion']
y_cls = df['congestion_class']

# Stratified split to maintain layout distribution
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42, stratify=df['tech_node']
)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=df['tech_node']
)

print(f"✅ Data split complete:")
print(f"   Training samples:   {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Testing samples:    {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
print(f"   Features:           {len(feature_cols)}")

# ============================================================
# STEP 5: TRAIN REGRESSION MODEL
# ============================================================
print("\n" + "="*70)
print("🤖 TRAINING REGRESSION MODEL (Random Forest)")
print("="*70)

# Initialize model with optimized hyperparameters
rf_reg = RandomForestRegressor(
    n_estimators=300,           # More trees = better accuracy
    max_depth=25,               # Deeper trees for complex patterns
    min_samples_split=5,        # Prevent overfitting
    min_samples_leaf=2,
    max_features='sqrt',        # Feature subsampling
    n_jobs=-1,                  # Use all CPU cores
    random_state=42,
    verbose=1                   # Show progress
)

print("\n🔄 Training started...")
rf_reg.fit(X_train, y_train_reg)
print("✅ Training complete!")

# Predictions
print("\n🔮 Generating predictions...")
y_pred_reg = rf_reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
mape = np.mean(np.abs((y_test_reg - y_pred_reg) / (y_test_reg + 1e-6))) * 100
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n" + "="*70)
print("📊 REGRESSION MODEL PERFORMANCE")
print("="*70)
print(f"\n   MSE:      {mse:.6f}")
print(f"   RMSE:     {rmse:.6f}")
print(f"   MAE:      {mae:.6f}")
print(f"   MAPE:     {mape:.3f}%")
print(f"   R² Score: {r2:.6f}")

# Cross-validation
print("\n🔄 Running 5-fold cross-validation...")
cv_scores = cross_val_score(rf_reg, X_train, y_train_reg, cv=5,
                             scoring='r2', n_jobs=-1)
print(f"   CV R² Scores: {cv_scores}")
print(f"   Mean CV R²:   {cv_scores.mean():.6f} (+/- {cv_scores.std()*2:.6f})")

# ============================================================
# STEP 6: TRAIN CLASSIFICATION MODEL
# ============================================================
print("\n" + "="*70)
print("🎯 TRAINING CLASSIFICATION MODEL (Random Forest)")
print("="*70)

rf_cls = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\n🔄 Training classifier...")
rf_cls.fit(X_train_cls, y_train_cls)
print("✅ Training complete!")

# Predictions
y_pred_cls = rf_cls.predict(X_test_cls)

# Evaluate
acc = accuracy_score(y_test_cls, y_pred_cls)

print("\n" + "="*70)
print("📊 CLASSIFICATION MODEL PERFORMANCE")
print("="*70)
print(f"\n   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n   Detailed Report:")
print(classification_report(y_test_cls, y_pred_cls))

# Confusion matrix
cm = confusion_matrix(y_test_cls, y_pred_cls, labels=['Green', 'Yellow', 'Red'])
print("\n   Confusion Matrix:")
print("   " + "          ".join(['Green', 'Yellow', 'Red']))
for i, label in enumerate(['Green', 'Yellow', 'Red']):
    print(f"   {label:7s} {cm[i]}")

# ============================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("\n" + "="*70)
print("📈 FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Get importances
importances = rf_reg.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n🏆 Top Features:")
print(feature_importance_df.to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
axes[0].barh(feature_importance_df['Feature'], feature_importance_df['Importance'],
             color='steelblue', edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Feature', fontsize=12, fontweight='bold')
axes[0].set_title('Feature Importance (Regression)', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(alpha=0.3, axis='x')

# Cumulative importance
cumulative = np.cumsum(feature_importance_df['Importance'].values)
axes[1].plot(range(1, len(cumulative)+1), cumulative, marker='o', linewidth=2, markersize=8)
axes[1].axhline(0.9, color='red', linestyle='--', linewidth=2, label='90% threshold')
axes[1].set_xlabel('Number of Features', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\n✅ Feature importance plot saved as 'feature_importance.png'")
plt.show()

# ============================================================
# STEP 8: PREDICTION QUALITY VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("📉 PREDICTION QUALITY ANALYSIS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Actual vs Predicted scatter
axes[0,0].scatter(y_test_reg, y_pred_reg, alpha=0.3, s=10, c='steelblue')
axes[0,0].plot([y_test_reg.min(), y_test_reg.max()],
               [y_test_reg.min(), y_test_reg.max()],
               'r--', linewidth=2, label='Perfect Prediction')
axes[0,0].set_xlabel('Actual Congestion', fontsize=11, fontweight='bold')
axes[0,0].set_ylabel('Predicted Congestion', fontsize=11, fontweight='bold')
axes[0,0].set_title(f'Actual vs Predicted (R²={r2:.4f})', fontsize=12, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# 2. Residual plot
residuals = y_test_reg - y_pred_reg
axes[0,1].scatter(y_pred_reg, residuals, alpha=0.3, s=10, c='coral')
axes[0,1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0,1].set_xlabel('Predicted Congestion', fontsize=11, fontweight='bold')
axes[0,1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0,1].set_title('Residual Plot (Should be Random)', fontsize=12, fontweight='bold')
axes[0,1].grid(alpha=0.3)

# 3. Error distribution
axes[1,0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1,0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1,0].set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
axes[1,0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1,0].set_title(f'Error Distribution (MAE={mae:.4f})', fontsize=12, fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# 4. Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1,1],
            xticklabels=['Green', 'Yellow', 'Red'],
            yticklabels=['Green', 'Yellow', 'Red'])
axes[1,1].set_xlabel('Predicted', fontsize=11, fontweight='bold')
axes[1,1].set_ylabel('Actual', fontsize=11, fontweight='bold')
axes[1,1].set_title(f'Confusion Matrix (Acc={acc:.4f})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('prediction_quality.png', dpi=150, bbox_inches='tight')
print("✅ Prediction quality plot saved as 'prediction_quality.png'")
plt.show()

# ============================================================
# STEP 9: PERFORMANCE BY TECH NODE
# ============================================================
print("\n" + "="*70)
print("🔬 PERFORMANCE BY TECHNOLOGY NODE")
print("="*70)

test_df = X_test.copy()
test_df['actual'] = y_test_reg.values
test_df['predicted'] = y_pred_reg
test_df['tech_node'] = df.iloc[X_test.index]['tech_node'].values

for tech in test_df['tech_node'].unique():
    tech_subset = test_df[test_df['tech_node'] == tech]
    tech_r2 = r2_score(tech_subset['actual'], tech_subset['predicted'])
    tech_mae = mean_absolute_error(tech_subset['actual'], tech_subset['predicted'])
    tech_mape = np.mean(np.abs((tech_subset['actual'] - tech_subset['predicted']) /
                                (tech_subset['actual'] + 1e-6))) * 100

    print(f"\n   {tech}:")
    print(f"      Samples: {len(tech_subset):,}")
    print(f"      R²:      {tech_r2:.6f}")
    print(f"      MAE:     {tech_mae:.6f}")
    print(f"      MAPE:    {tech_mape:.3f}%")

# ============================================================
# STEP 10: SAVE MODELS
# ============================================================
print("\n" + "="*70)
print("💾 SAVING MODELS")
print("="*70)

import pickle

# Save regression model
with open('congestion_regression_rf.pkl', 'wb') as f:
    pickle.dump(rf_reg, f)
print("✅ Regression model saved: congestion_regression_rf.pkl")

# Save classification model
with open('congestion_classification_rf.pkl', 'wb') as f:
    pickle.dump(rf_cls, f)
print("✅ Classification model saved: congestion_classification_rf.pkl")

# Save feature names and encoder
with open('feature_config.pkl', 'wb') as f:
    pickle.dump({
        'feature_cols': feature_cols,
        'label_encoder': le
    }, f)
print("✅ Feature config saved: feature_config.pkl")

# ============================================================
# STEP 11: TEST PREDICTIONS ON SAMPLE DATA
# ============================================================
print("\n" + "="*70)
print("🔮 TESTING ON SAMPLE DESIGNS")
print("="*70)

# Create 5 sample designs
samples = pd.DataFrame([
    # Low congestion design
    {
        'density': 0.3, 'pin_density': 0.5, 'fanout': 3.2, 'macro': 0,
        'capacity': 7.5, 'macro_proximity': 0.1, 'wire_length': 50,
        'clock_region': 0, 'drc_violations': 0, 'net_criticality': 0.2,
        'power_density': 0.3, 'temperature': 0.3, 'tech_node_encoded': 2  # 7nm
    },
    # Medium congestion design
    {
        'density': 0.6, 'pin_density': 1.2, 'fanout': 6.5, 'macro': 0,
        'capacity': 5.0, 'macro_proximity': 0.3, 'wire_length': 120,
        'clock_region': 0.5, 'drc_violations': 0.2, 'net_criticality': 0.5,
        'power_density': 0.6, 'temperature': 0.6, 'tech_node_encoded': 1  # 14nm
    },
    # High congestion design
    {
        'density': 0.9, 'pin_density': 2.5, 'fanout': 12.0, 'macro': 1,
        'capacity': 2.0, 'macro_proximity': 0.8, 'wire_length': 250,
        'clock_region': 1.0, 'drc_violations': 0.6, 'net_criticality': 0.9,
        'power_density': 0.9, 'temperature': 0.9, 'tech_node_encoded': 0  # 28nm
    },
    # Clock-heavy design
    {
        'density': 0.5, 'pin_density': 1.0, 'fanout': 80.0, 'macro': 0,
        'capacity': 6.0, 'macro_proximity': 0.2, 'wire_length': 100,
        'clock_region': 1.0, 'drc_violations': 0.1, 'net_criticality': 0.8,
        'power_density': 0.5, 'temperature': 0.5, 'tech_node_encoded': 2  # 7nm
    },
    # Macro-dense design
    {
        'density': 0.4, 'pin_density': 1.8, 'fanout': 5.0, 'macro': 1,
        'capacity': 1.5, 'macro_proximity': 0.9, 'wire_length': 80,
        'clock_region': 0.2, 'drc_violations': 0.4, 'net_criticality': 0.4,
        'power_density': 0.7, 'temperature': 0.6, 'tech_node_encoded': 1  # 14nm
    }
])

# Predict
pred_reg = rf_reg.predict(samples)
pred_cls = rf_cls.predict(samples)

print("\n🔍 Sample Predictions:\n")
design_types = ['Low Congestion', 'Medium Congestion', 'High Congestion',
                'Clock-Heavy', 'Macro-Dense']

for i, design_type in enumerate(design_types):
    print(f"{'='*50}")
    print(f"Design {i+1}: {design_type}")
    print(f"{'='*50}")
    print(f"   Predicted Congestion: {pred_reg[i]:.4f}")
    print(f"   Predicted Class:      {pred_cls[i]}")
    print(f"   Key Features:")
    print(f"      Density:        {samples.iloc[i]['density']:.2f}")
    print(f"      Fanout:         {samples.iloc[i]['fanout']:.1f}")
    print(f"      Capacity:       {samples.iloc[i]['capacity']:.1f}")
    print(f"      Clock Region:   {samples.iloc[i]['clock_region']:.1f}")
    print(f"      DRC Violations: {samples.iloc[i]['drc_violations']:.1f}")
    print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("✅ TRAINING COMPLETE - SUMMARY")
print("="*70)

summary = f"""
📊 DATASET:
   Total samples:        {len(df):,}
   Training samples:     {len(X_train):,}
   Testing samples:      {len(X_test):,}
   Features:             {len(feature_cols)}
   Technology nodes:     {df['tech_node'].nunique()}
   Layouts:              {df['layout_id'].nunique()}

🤖 REGRESSION MODEL:
   R² Score:             {r2:.6f}
   RMSE:                 {rmse:.6f}
   MAE:                  {mae:.6f}
   MAPE:                 {mape:.3f}%
   Cross-Val R²:         {cv_scores.mean():.6f}

🎯 CLASSIFICATION MODEL:
   Accuracy:             {acc:.4f} ({acc*100:.2f}%)
   Classes:              Green, Yellow, Red

💾 SAVED FILES:
   • congestion_regression_rf.pkl
   • congestion_classification_rf.pkl
   • feature_config.pkl
   • congestion_distribution.png
   • feature_importance.png
   • prediction_quality.png

🏆 TOP 3 FEATURES:
   1. {feature_importance_df.iloc[0]['Feature']:20s} ({feature_importance_df.iloc[0]['Importance']:.4f})
   2. {feature_importance_df.iloc[1]['Feature']:20s} ({feature_importance_df.iloc[1]['Importance']:.4f})
   3. {feature_importance_df.iloc[2]['Feature']:20s} ({feature_importance_df.iloc[2]['Importance']:.4f})
"""

print(summary)
print("="*70)
print("🎉 ALL DONE! Models are ready for deployment.")
print("="*70)