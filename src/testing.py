import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("🧪 COMPREHENSIVE TESTING PIPELINE - UNSEEN LAYOUTS")
print("="*70)

# ============================================================
# EMBEDDED GENERATOR CLASS (for standalone testing)
# ============================================================

class RealisticCongestionGenerator:
    """Generates industry-standard routing congestion datasets"""

    def __init__(self, grid_size=64, num_layouts=50):
        self.N = grid_size
        self.num_layouts = num_layouts
        self.DESIGN_RULES = {
            'min_macro_spacing': 5,
            'routing_layers': 6,
            'tracks_per_layer': 10,
            'wire_pitch_nm': 48,
            'via_resistance': 0.1,
            'clock_tree_overhead': 0.15,
        }

    def generate_realistic_layout(self, seed=42, tech_node='7nm'):
        np.random.seed(seed)
        tech_params = self._get_tech_params(tech_node)
        macros, macro_coords = self._place_macros_with_rules()
        density = self._generate_cell_density()

        macro_pins = macros * np.random.uniform(10, 50, (self.N, self.N))
        cell_pins = density * np.random.uniform(2, 6, (self.N, self.N))
        pin_density = macro_pins + cell_pins

        base_fanout = density * np.random.uniform(2.5, 8.5, (self.N, self.N))
        clock_regions = self._generate_clock_tree_regions()
        clock_fanout = clock_regions * np.random.uniform(50, 200, (self.N, self.N))
        fanout = base_fanout + clock_fanout

        wire_length = self._generate_wire_lengths(density, macro_coords)

        base_capacity = tech_params['base_capacity']
        capacity = np.full((self.N, self.N), base_capacity)
        capacity -= macros * (base_capacity * 0.8)
        capacity -= (density > 0.7) * (base_capacity * 0.3)
        capacity = np.clip(capacity, base_capacity * 0.1, None)

        macro_prox = self._compute_macro_proximity(macro_coords)

        base_demand = (
            0.3 * density +
            0.2 * pin_density / 10 +
            0.15 * (fanout / 10) +
            0.1 * (wire_length / 100)
        )
        clock_demand = clock_regions * self.DESIGN_RULES['clock_tree_overhead']
        macro_effect = macro_prox * 0.25
        total_demand = base_demand + clock_demand + macro_effect

        drc_violations = self._detect_drc_hotspots(density, macros)
        total_demand += drc_violations * 0.2

        congestion = total_demand / capacity
        overflow_mask = congestion > 1.0
        congestion[overflow_mask] = 1.0 + (congestion[overflow_mask] - 1.0) ** 1.5
        congestion = congestion.clip(0, 5)
        congestion = gaussian_filter(congestion, sigma=1.0)

        criticality = self._compute_net_criticality(density, clock_regions)
        power_density = self._generate_power_grid(macros, density)
        temperature = self._compute_thermal_map(density, macros, fanout)

        df = pd.DataFrame({
            'density': density.flatten(),
            'pin_density': pin_density.flatten(),
            'fanout': fanout.flatten(),
            'macro': macros.flatten(),
            'capacity': capacity.flatten(),
            'macro_proximity': macro_prox.flatten(),
            'wire_length': wire_length.flatten(),
            'clock_region': clock_regions.flatten(),
            'drc_violations': drc_violations.flatten(),
            'net_criticality': criticality.flatten(),
            'power_density': power_density.flatten(),
            'temperature': temperature.flatten(),
            'congestion': congestion.flatten()
        })

        df['tech_node'] = tech_node
        df['layout_id'] = seed

        return df, congestion

    def _get_tech_params(self, tech_node):
        params = {
            '28nm':  {'base_capacity': 5.0, 'wire_resistance': 0.25},
            '14nm':  {'base_capacity': 6.0, 'wire_resistance': 0.3},
            '7nm':   {'base_capacity': 8.0, 'wire_resistance': 0.35},
        }
        return params.get(tech_node, params['7nm'])

    def _place_macros_with_rules(self):
        macros = np.zeros((self.N, self.N))
        macro_coords = []
        num_macros = np.random.randint(4, 10)
        min_spacing = self.DESIGN_RULES['min_macro_spacing']

        for _ in range(num_macros):
            placed = False
            attempts = 0
            while not placed and attempts < 50:
                cx = np.random.randint(8, self.N-8)
                cy = np.random.randint(8, self.N-8)
                valid = True
                for prev_cx, prev_cy in macro_coords:
                    if np.sqrt((cx-prev_cx)**2 + (cy-prev_cy)**2) < min_spacing:
                        valid = False
                        break
                if valid:
                    size = np.random.randint(3, 7)
                    macros[cx-size:cx+size, cy-size:cy+size] = 1
                    macro_coords.append((cx, cy))
                    placed = True
                attempts += 1
        return macros, macro_coords

    def _generate_cell_density(self):
        density = np.zeros((self.N, self.N))
        num_clusters = np.random.randint(15, 25)
        for _ in range(num_clusters):
            cx, cy = np.random.randint(0, self.N), np.random.randint(0, self.N)
            x, y = np.arange(self.N), np.arange(self.N)
            xv, yv = np.meshgrid(x, y)
            spread = np.random.uniform(50, 150)
            intensity = np.random.uniform(0.5, 1.5)
            blob = intensity * np.exp(-((xv-cx)**2 + (yv-cy)**2) / spread)
            density += blob
        density = density / density.max()
        return np.clip(density, 0, 1)

    def _generate_clock_tree_regions(self):
        clock = np.zeros((self.N, self.N))
        mid = self.N // 2
        clock[mid-2:mid+2, :] = 1
        clock[:, mid-2:mid+2] = 1
        clock[mid//2-1:mid//2+1, :] = 0.5
        clock[3*mid//2-1:3*mid//2+1, :] = 0.5
        return clock

    def _generate_wire_lengths(self, density, macro_coords):
        wire_length = density * np.random.uniform(10, 50, (self.N, self.N))
        for cx, cy in macro_coords:
            x, y = np.arange(self.N), np.arange(self.N)
            xv, yv = np.meshgrid(x, y)
            wire_length += np.sqrt((xv-cx)**2 + (yv-cy)**2) * 2
        return np.clip(wire_length, 10, 500)

    def _compute_macro_proximity(self, macro_coords):
        macro_prox = np.zeros((self.N, self.N))
        for x in range(self.N):
            for y in range(self.N):
                if len(macro_coords) > 0:
                    min_dist = min([np.sqrt((x-cx)**2 + (y-cy)**2) for cx, cy in macro_coords])
                    macro_prox[x, y] = 1 / (min_dist + 1)
        return macro_prox

    def _detect_drc_hotspots(self, density, macros):
        kernel = np.ones((5, 5)) / 25
        crowding = convolve2d(density, kernel, mode='same')
        drc = (crowding > 0.6).astype(float) * 0.5
        drc += (macros > 0).astype(float) * 0.3
        return drc

    def _compute_net_criticality(self, density, clock_regions):
        return np.clip(density * 0.5 + clock_regions * 0.8, 0, 1)

    def _generate_power_grid(self, macros, density):
        power = macros * 0.6 + density * 0.4
        return gaussian_filter(power, sigma=2.0)

    def _compute_thermal_map(self, density, macros, fanout):
        temperature = density * 0.4 + macros * 0.3 + (fanout / fanout.max()) * 0.3
        return gaussian_filter(temperature, sigma=3.0)

# ============================================================
# STEP 1: LOAD TRAINED MODELS
# ============================================================
print("\n📂 Loading trained models...")

try:
    with open('congestion_regression_rf.pkl', 'rb') as f:
        rf_reg = pickle.load(f)
    print("✅ Regression model loaded")

    with open('congestion_classification_rf.pkl', 'rb') as f:
        rf_cls = pickle.load(f)
    print("✅ Classification model loaded")

    with open('feature_config.pkl', 'rb') as f:
        config = pickle.load(f)
        feature_cols = config['feature_cols']
        le = config['label_encoder']
    print("✅ Feature configuration loaded")

except FileNotFoundError as e:
    print(f"❌ Error: Required model files not found!")
    print("   Please run the training pipeline first.")
    exit()

# ============================================================
# STEP 2: GENERATE NEW TEST LAYOUTS (UNSEEN DATA)
# ============================================================
print("\n" + "="*70)
print("🔬 GENERATING NEW TEST LAYOUTS (NEVER SEEN BEFORE)")
print("="*70)

test_generator = RealisticCongestionGenerator(grid_size=64, num_layouts=5)

test_scenarios = [
    {'seed': 9000, 'tech_node': '7nm', 'name': 'Test Case 1: 7nm Low-Power'},
    {'seed': 9001, 'tech_node': '14nm', 'name': 'Test Case 2: 14nm Balanced'},
    {'seed': 9002, 'tech_node': '28nm', 'name': 'Test Case 3: 28nm High-Density'},
    {'seed': 9003, 'tech_node': '7nm', 'name': 'Test Case 4: 7nm Clock-Intensive'},
    {'seed': 9004, 'tech_node': '14nm', 'name': 'Test Case 5: 14nm Macro-Heavy'},
]

print(f"\n🔄 Generating {len(test_scenarios)} new test layouts...")
print("   These layouts have NEVER been seen by the model during training!\n")

test_data = []
test_congestion_maps = []
test_names = []

for scenario in test_scenarios:
    print(f"   Generating: {scenario['name']}...")
    df, cong_map = test_generator.generate_realistic_layout(
        seed=scenario['seed'],
        tech_node=scenario['tech_node']
    )
    test_data.append(df)
    test_congestion_maps.append(cong_map)
    test_names.append(scenario['name'])

print("\n✅ Test data generation complete!")

# ============================================================
# STEP 3: PREPARE TEST DATA FOR PREDICTION
# ============================================================
print("\n" + "="*70)
print("📊 PREPARING TEST DATA")
print("="*70)

test_df = pd.concat(test_data, ignore_index=True)
test_df['tech_node_encoded'] = le.transform(test_df['tech_node'])

X_test = test_df[feature_cols]
y_test_actual = test_df['congestion']

print(f"\n   Test samples: {len(X_test):,}")
print(f"   Test layouts: {len(test_scenarios)}")
print(f"   Samples per layout: {len(X_test) // len(test_scenarios):,}")

# ============================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================
print("\n" + "="*70)
print("🔮 RUNNING PREDICTIONS ON TEST DATA")
print("="*70)

print("\n🔄 Predicting congestion values...")
y_test_pred = rf_reg.predict(X_test)

print("🔄 Predicting congestion classes...")
def classify_congestion(c):
    if c < 0.8:
        return "Green"
    elif c < 1.5:
        return "Yellow"
    else:
        return "Red"

y_test_class_actual = test_df['congestion'].apply(classify_congestion)
y_test_class_pred = rf_cls.predict(X_test)

print("✅ Predictions complete!")

# ============================================================
# STEP 5: EVALUATE OVERALL PERFORMANCE
# ============================================================
print("\n" + "="*70)
print("📈 OVERALL TEST PERFORMANCE")
print("="*70)

mse = mean_squared_error(y_test_actual, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_test_pred)
mape = np.mean(np.abs((y_test_actual - y_test_pred) / (y_test_actual + 1e-6))) * 100
r2 = r2_score(y_test_actual, y_test_pred)

print("\n🤖 REGRESSION METRICS:")
print(f"   R² Score:  {r2:.6f}")
print(f"   RMSE:      {rmse:.6f}")
print(f"   MAE:       {mae:.6f}")
print(f"   MAPE:      {mape:.3f}%")

acc = accuracy_score(y_test_class_actual, y_test_class_pred)
print(f"\n🎯 CLASSIFICATION ACCURACY: {acc:.4f} ({acc*100:.2f}%)")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test_class_actual, y_test_class_pred))

cm = confusion_matrix(y_test_class_actual, y_test_class_pred,
                      labels=['Green', 'Yellow', 'Red'])
print("\n📊 Confusion Matrix:")
print("             Predicted")
print("             Green  Yellow  Red")
for i, label in enumerate(['Green', 'Yellow', 'Red']):
    print(f"Actual {label:6s}  {cm[i]}")

# ============================================================
# STEP 6: PER-LAYOUT ANALYSIS
# ============================================================
print("\n" + "="*70)
print("🔬 PER-LAYOUT TEST RESULTS")
print("="*70)

samples_per_layout = len(X_test) // len(test_scenarios)
layout_results = []

for i, (name, gt_map) in enumerate(zip(test_names, test_congestion_maps)):
    start_idx = i * samples_per_layout
    end_idx = (i + 1) * samples_per_layout

    actual = y_test_actual.iloc[start_idx:end_idx]
    predicted = y_test_pred[start_idx:end_idx]
    pred_map = predicted.reshape(64, 64)

    layout_mse = mean_squared_error(actual, predicted)
    layout_rmse = np.sqrt(layout_mse)
    layout_mae = mean_absolute_error(actual, predicted)
    layout_r2 = r2_score(actual, predicted)
    layout_mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100

    pixel_error = np.abs(gt_map.flatten() - predicted)
    max_error = pixel_error.max()

    layout_results.append({
        'name': name,
        'r2': layout_r2,
        'rmse': layout_rmse,
        'mae': layout_mae,
        'mape': layout_mape,
        'max_error': max_error,
        'pred_map': pred_map,
        'gt_map': gt_map
    })

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"   R² Score:     {layout_r2:.6f}")
    print(f"   RMSE:         {layout_rmse:.6f}")
    print(f"   MAE:          {layout_mae:.6f}")
    print(f"   MAPE:         {layout_mape:.3f}%")
    print(f"   Max Error:    {max_error:.6f}")
    print(f"   Actual Mean:  {gt_map.mean():.4f}")
    print(f"   Pred Mean:    {pred_map.mean():.4f}")

# ============================================================
# STEP 7: VISUALIZATION - PREDICTION vs GROUND TRUTH
# ============================================================
print("\n" + "="*70)
print("🎨 CREATING COMPARISON VISUALIZATIONS")
print("="*70)

colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
cmap_custom = LinearSegmentedColormap.from_list('congestion', colors, N=100)

fig, axes = plt.subplots(5, 3, figsize=(15, 20))

for i, result in enumerate(layout_results):
    gt_map = result['gt_map']
    pred_map = result['pred_map']
    error_map = np.abs(gt_map - pred_map)

    im1 = axes[i, 0].imshow(gt_map, cmap=cmap_custom, vmin=0, vmax=3, origin='lower')
    axes[i, 0].set_title(f'Ground Truth\n{result["name"]}', fontsize=10, fontweight='bold')
    axes[i, 0].set_ylabel('Y', fontsize=9)
    axes[i, 0].set_xlabel('X', fontsize=9)
    plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)

    im2 = axes[i, 1].imshow(pred_map, cmap=cmap_custom, vmin=0, vmax=3, origin='lower')
    axes[i, 1].set_title(f'Predicted\nR²={result["r2"]:.4f}', fontsize=10, fontweight='bold')
    axes[i, 1].set_ylabel('Y', fontsize=9)
    axes[i, 1].set_xlabel('X', fontsize=9)
    plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)

    im3 = axes[i, 2].imshow(error_map, cmap='Reds', vmin=0, vmax=1, origin='lower')
    axes[i, 2].set_title(f'Absolute Error\nMAE={result["mae"]:.4f}',
                         fontsize=10, fontweight='bold')
    axes[i, 2].set_ylabel('Y', fontsize=9)
    axes[i, 2].set_xlabel('X', fontsize=9)
    plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)

plt.suptitle('Test Results: Ground Truth vs Predictions vs Error',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('test_results_comparison.png', dpi=200, bbox_inches='tight')
print("✅ Comparison plot saved: test_results_comparison.png")
plt.show()

# ============================================================
# STEP 8: SCATTER PLOT - ACTUAL VS PREDICTED
# ============================================================
print("\n" + "="*70)
print("📊 CREATING SCATTER ANALYSIS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].scatter(y_test_actual, y_test_pred, alpha=0.1, s=5, c='steelblue')
axes[0, 0].plot([0, 3], [0, 3], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Congestion', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Congestion', fontsize=11, fontweight='bold')
axes[0, 0].set_title(f'Overall: R²={r2:.4f}', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

residuals = y_test_actual - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.1, s=5, c='coral')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Congestion', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0, 2].set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
axes[0, 2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 2].set_title(f'Error Distribution (MAE={mae:.4f})', fontsize=12, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1, 0],
            xticklabels=['Green', 'Yellow', 'Red'],
            yticklabels=['Green', 'Yellow', 'Red'])
axes[1, 0].set_xlabel('Predicted', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Actual', fontsize=11, fontweight='bold')
axes[1, 0].set_title(f'Confusion Matrix (Acc={acc:.4f})', fontsize=12, fontweight='bold')

layout_names_short = [f"Test {i+1}" for i in range(len(layout_results))]
layout_r2s = [r['r2'] for r in layout_results]
axes[1, 1].bar(layout_names_short, layout_r2s, color='steelblue', edgecolor='black')
axes[1, 1].axhline(0.9, color='red', linestyle='--', linewidth=2, label='Target: 0.9')
axes[1, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Per-Layout R² Scores', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')
axes[1, 1].tick_params(axis='x', rotation=45)

layout_maes = [r['mae'] for r in layout_results]
axes[1, 2].bar(layout_names_short, layout_maes, color='coral', edgecolor='black')
axes[1, 2].set_ylabel('MAE', fontsize=11, fontweight='bold')
axes[1, 2].set_title('Per-Layout Mean Absolute Error', fontsize=12, fontweight='bold')
axes[1, 2].grid(alpha=0.3, axis='y')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('test_analysis_metrics.png', dpi=200, bbox_inches='tight')
print("✅ Analysis plot saved: test_analysis_metrics.png")
plt.show()

# ============================================================
# STEP 9: SAVE TEST RESULTS
# ============================================================
print("\n" + "="*70)
print("💾 SAVING TEST RESULTS")
print("="*70)

summary_data = []
for result in layout_results:
    summary_data.append({
        'Test_Case': result['name'],
        'R2_Score': result['r2'],
        'RMSE': result['rmse'],
        'MAE': result['mae'],
        'MAPE': result['mape'],
        'Max_Error': result['max_error']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('test_results_summary.csv', index=False)
print("✅ Test summary saved: test_results_summary.csv")

test_df['predicted_congestion'] = y_test_pred
test_df['prediction_error'] = np.abs(test_df['congestion'] - y_test_pred)
test_df.to_csv('test_predictions_detailed.csv', index=False)
print("✅ Detailed predictions saved: test_predictions_detailed.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("✅ COMPREHENSIVE TESTING COMPLETE")
print("="*70)

final_summary = f"""
📊 TEST DATASET:
   Test layouts:         {len(test_scenarios)}
   Total test samples:   {len(X_test):,}
   Samples per layout:   {samples_per_layout:,}

🎯 OVERALL PERFORMANCE:
   R² Score:             {r2:.6f}
   RMSE:                 {rmse:.6f}
   MAE:                  {mae:.6f}
   MAPE:                 {mape:.3f}%
   Classification Acc:   {acc:.4f} ({acc*100:.2f}%)

🏆 BEST TEST CASE:
   Layout:               {layout_results[np.argmax(layout_r2s)]['name']}
   R² Score:             {max(layout_r2s):.6f}

⚠️  WORST TEST CASE:
   Layout:               {layout_results[np.argmin(layout_r2s)]['name']}
   R² Score:             {min(layout_r2s):.6f}

💾 OUTPUT FILES:
   • test_results_comparison.png
   • test_analysis_metrics.png
   • test_results_summary.csv
   • test_predictions_detailed.csv

📈 INTERPRETATION:
   R² > 0.90:  Excellent prediction accuracy
   R² > 0.80:  Good prediction accuracy
   R² > 0.70:  Acceptable prediction accuracy
   R² < 0.70:  Needs improvement

   Current Score: {r2:.4f} - {'✅ EXCELLENT!' if r2 > 0.9 else '✓ GOOD' if r2 > 0.8 else '⚠️ ACCEPTABLE' if r2 > 0.7 else '❌ NEEDS WORK'}
"""

print(final_summary)
print("="*70)
print("🎉 Testing pipeline complete!")
print("="*70)