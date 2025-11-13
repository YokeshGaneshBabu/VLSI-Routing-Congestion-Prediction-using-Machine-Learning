import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("🗺️  SYNTHETIC CONGESTION MAP GENERATOR")
print("="*70)

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
# STEP 2: CONGESTION MAP GENERATOR CLASS
# ============================================================

class SyntheticCongestionMapGenerator:
    """Generate realistic congestion heatmaps from ML predictions"""

    def __init__(self, model, feature_cols, grid_size=64):
        self.model = model
        self.feature_cols = feature_cols
        self.grid_size = grid_size

    def create_spatial_features(self, grid_size, design_params):
        """Create spatially-varying features across the chip"""

        features_grid = {}

        # Generate spatial patterns for each feature
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)

        # Density: Radial gradient (higher in center)
        center_x, center_y = design_params.get('center', (0.5, 0.5))
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        base_density = design_params.get('base_density', 0.5)
        features_grid['density'] = base_density * (1 + 0.4 * (1 - dist_from_center))
        features_grid['density'] = np.clip(features_grid['density'], 0, 1)

        # Pin density: Correlated with density but noisier
        features_grid['pin_density'] = features_grid['density'] * \
            design_params.get('pin_factor', 1.5) + \
            np.random.normal(0, 0.1, (grid_size, grid_size))
        features_grid['pin_density'] = np.clip(features_grid['pin_density'], 0, 3)

        # Fanout: Higher in certain regions
        fanout_hotspots = design_params.get('fanout_hotspots', 3)
        features_grid['fanout'] = np.ones((grid_size, grid_size)) * 4.0
        for _ in range(fanout_hotspots):
            hx, hy = np.random.uniform(0.2, 0.8, 2)
            hotspot = np.exp(-20 * ((X - hx)**2 + (Y - hy)**2))
            features_grid['fanout'] += hotspot * np.random.uniform(5, 15)

        # Macro presence: Discrete blocks
        features_grid['macro'] = np.zeros((grid_size, grid_size))
        num_macros = design_params.get('num_macros', 2)
        for _ in range(num_macros):
            mx, my = np.random.randint(5, grid_size-15, 2)
            mw, mh = np.random.randint(8, 15, 2)
            features_grid['macro'][my:my+mh, mx:mx+mw] = 1

        # Capacity: Inversely related to density
        base_capacity = design_params.get('base_capacity', 6.0)
        features_grid['capacity'] = base_capacity * (1.2 - features_grid['density'] * 0.8)
        features_grid['capacity'] = np.clip(features_grid['capacity'], 1, 10)

        # Macro proximity: Distance from nearest macro
        if num_macros > 0:
            from scipy.ndimage import distance_transform_edt
            features_grid['macro_proximity'] = 1 - distance_transform_edt(
                1 - features_grid['macro']
            ) / (grid_size / 4)
            features_grid['macro_proximity'] = np.clip(features_grid['macro_proximity'], 0, 1)
        else:
            features_grid['macro_proximity'] = np.zeros((grid_size, grid_size))

        # Wire length: Related to density and distance
        features_grid['wire_length'] = features_grid['density'] * 200 + \
            np.random.normal(0, 20, (grid_size, grid_size))
        features_grid['wire_length'] = np.clip(features_grid['wire_length'], 10, 300)

        # Clock regions: Vertical stripes
        clock_intensity = design_params.get('clock_intensity', 0.5)
        features_grid['clock_region'] = clock_intensity * \
            (0.5 + 0.5 * np.sin(X * np.pi * 4))
        features_grid['clock_region'] = np.clip(features_grid['clock_region'], 0, 1)

        # DRC violations: Sparse hotspots
        features_grid['drc_violations'] = np.random.exponential(
            0.1, (grid_size, grid_size)
        ) * (features_grid['density'] > 0.7).astype(float)
        features_grid['drc_violations'] = np.clip(features_grid['drc_violations'], 0, 1)

        # Net criticality: Higher near clock regions
        features_grid['net_criticality'] = 0.3 + 0.5 * features_grid['clock_region'] + \
            np.random.normal(0, 0.1, (grid_size, grid_size))
        features_grid['net_criticality'] = np.clip(features_grid['net_criticality'], 0, 1)

        # Power density: Correlated with density
        features_grid['power_density'] = features_grid['density'] * 0.9 + \
            np.random.normal(0, 0.05, (grid_size, grid_size))
        features_grid['power_density'] = np.clip(features_grid['power_density'], 0, 1)

        # Temperature: Smoothed power density
        features_grid['temperature'] = gaussian_filter(features_grid['power_density'], sigma=2)
        features_grid['temperature'] = np.clip(features_grid['temperature'], 0, 1)

        # Tech node (constant across chip)
        tech_node_value = design_params.get('tech_node_encoded', 2)
        features_grid['tech_node_encoded'] = np.full((grid_size, grid_size), tech_node_value)

        return features_grid

    def generate_congestion_map(self, design_params):
        """Generate a congestion map using ML predictions"""

        # Create spatial features
        features_grid = self.create_spatial_features(self.grid_size, design_params)

        # Prepare features for prediction (flatten grid to samples)
        n_samples = self.grid_size * self.grid_size
        X_pred = np.zeros((n_samples, len(self.feature_cols)))

        for i, feat_name in enumerate(self.feature_cols):
            X_pred[:, i] = features_grid[feat_name].flatten()

        # Predict congestion
        congestion_flat = self.model.predict(X_pred)
        congestion_map = congestion_flat.reshape(self.grid_size, self.grid_size)

        # Apply spatial smoothing for realism
        congestion_map = gaussian_filter(congestion_map, sigma=1.5)

        return congestion_map, features_grid

# ============================================================
# STEP 3: GENERATE MULTIPLE CONGESTION MAPS
# ============================================================

print("\n" + "="*70)
print("🎨 GENERATING SYNTHETIC CONGESTION MAPS")
print("="*70)

generator = SyntheticCongestionMapGenerator(rf_reg, feature_cols, grid_size=64)

# Define different design scenarios
design_scenarios = [
    {
        'name': 'Low Congestion Design (7nm)',
        'base_density': 0.3,
        'pin_factor': 1.0,
        'fanout_hotspots': 1,
        'num_macros': 1,
        'base_capacity': 8.0,
        'clock_intensity': 0.2,
        'tech_node_encoded': 2,  # 7nm
        'center': (0.5, 0.5)
    },
    {
        'name': 'Medium Congestion Design (14nm)',
        'base_density': 0.6,
        'pin_factor': 1.5,
        'fanout_hotspots': 3,
        'num_macros': 2,
        'base_capacity': 5.0,
        'clock_intensity': 0.5,
        'tech_node_encoded': 1,  # 14nm
        'center': (0.6, 0.4)
    },
    {
        'name': 'High Congestion Design (28nm)',
        'base_density': 0.85,
        'pin_factor': 2.0,
        'fanout_hotspots': 5,
        'num_macros': 3,
        'base_capacity': 2.5,
        'clock_intensity': 0.8,
        'tech_node_encoded': 0,  # 28nm
        'center': (0.5, 0.5)
    },
    {
        'name': 'Clock-Heavy Design (7nm)',
        'base_density': 0.5,
        'pin_factor': 1.2,
        'fanout_hotspots': 2,
        'num_macros': 1,
        'base_capacity': 6.0,
        'clock_intensity': 1.0,
        'tech_node_encoded': 2,  # 7nm
        'center': (0.5, 0.5)
    },
    {
        'name': 'Macro-Dense Design (14nm)',
        'base_density': 0.4,
        'pin_factor': 1.8,
        'fanout_hotspots': 2,
        'num_macros': 5,
        'base_capacity': 4.0,
        'clock_intensity': 0.3,
        'tech_node_encoded': 1,  # 14nm
        'center': (0.5, 0.5)
    },
    {
        'name': 'Edge-Heavy Design (7nm)',
        'base_density': 0.55,
        'pin_factor': 1.4,
        'fanout_hotspots': 4,
        'num_macros': 2,
        'base_capacity': 5.5,
        'clock_intensity': 0.4,
        'tech_node_encoded': 2,  # 7nm
        'center': (0.2, 0.8)
    }
]

# Generate maps for each scenario
congestion_maps = []
feature_grids = []

print("\n🔄 Generating maps...")
for i, scenario in enumerate(design_scenarios):
    print(f"   [{i+1}/{len(design_scenarios)}] {scenario['name']}")
    cmap, fgrid = generator.generate_congestion_map(scenario)
    congestion_maps.append(cmap)
    feature_grids.append(fgrid)

print("✅ All maps generated!")

# ============================================================
# STEP 4: VISUALIZE CONGESTION MAPS
# ============================================================

print("\n" + "="*70)
print("📊 CREATING VISUALIZATIONS")
print("="*70)

# Custom colormap (Green -> Yellow -> Red)
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
n_bins = 100
cmap_custom = LinearSegmentedColormap.from_list('congestion', colors, N=n_bins)

# Plot all congestion maps
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (cmap, scenario) in enumerate(zip(congestion_maps, design_scenarios)):
    im = axes[i].imshow(cmap, cmap=cmap_custom, interpolation='bilinear',
                        vmin=0, vmax=3, origin='lower')
    axes[i].set_title(scenario['name'], fontsize=13, fontweight='bold', pad=10)
    axes[i].set_xlabel('X Position', fontsize=10)
    axes[i].set_ylabel('Y Position', fontsize=10)
    axes[i].grid(False)

    # Add statistics overlay
    mean_cong = cmap.mean()
    max_cong = cmap.max()
    critical_area = (cmap > 1.5).sum() / cmap.size * 100

    stats_text = f"Mean: {mean_cong:.2f}\nMax: {max_cong:.2f}\nCritical: {critical_area:.1f}%"
    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    cbar.set_label('Congestion', fontsize=9)

plt.tight_layout()
plt.savefig('synthetic_congestion_maps.png', dpi=200, bbox_inches='tight')
print("✅ Congestion maps saved as 'synthetic_congestion_maps.png'")
plt.show()

# ============================================================
# STEP 5: DETAILED ANALYSIS OF ONE MAP
# ============================================================

print("\n" + "="*70)
print("🔬 DETAILED ANALYSIS - HIGH CONGESTION DESIGN")
print("="*70)

# Analyze the high congestion design (index 2)
analysis_idx = 2
cmap_analysis = congestion_maps[analysis_idx]
fgrid_analysis = feature_grids[analysis_idx]

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

# 1. Congestion map
im1 = axes[0,0].imshow(cmap_analysis, cmap=cmap_custom, vmin=0, vmax=3, origin='lower')
axes[0,0].set_title('Predicted Congestion', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0,0], fraction=0.046)

# 2. Density
im2 = axes[0,1].imshow(fgrid_analysis['density'], cmap='viridis', origin='lower')
axes[0,1].set_title('Cell Density', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[0,1], fraction=0.046)

# 3. Pin Density
im3 = axes[0,2].imshow(fgrid_analysis['pin_density'], cmap='plasma', origin='lower')
axes[0,2].set_title('Pin Density', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=axes[0,2], fraction=0.046)

# 4. Fanout
im4 = axes[1,0].imshow(fgrid_analysis['fanout'], cmap='hot', origin='lower')
axes[1,0].set_title('Fanout Distribution', fontsize=12, fontweight='bold')
plt.colorbar(im4, ax=axes[1,0], fraction=0.046)

# 5. Macros
im5 = axes[1,1].imshow(fgrid_analysis['macro'], cmap='binary', origin='lower')
axes[1,1].set_title('Macro Placement', fontsize=12, fontweight='bold')
plt.colorbar(im5, ax=axes[1,1], fraction=0.046)

# 6. Capacity
im6 = axes[1,2].imshow(fgrid_analysis['capacity'], cmap='coolwarm_r', origin='lower')
axes[1,2].set_title('Routing Capacity', fontsize=12, fontweight='bold')
plt.colorbar(im6, ax=axes[1,2], fraction=0.046)

# 7. Clock Regions
im7 = axes[2,0].imshow(fgrid_analysis['clock_region'], cmap='copper', origin='lower')
axes[2,0].set_title('Clock Regions', fontsize=12, fontweight='bold')
plt.colorbar(im7, ax=axes[2,0], fraction=0.046)

# 8. Temperature
im8 = axes[2,1].imshow(fgrid_analysis['temperature'], cmap='inferno', origin='lower')
axes[2,1].set_title('Temperature', fontsize=12, fontweight='bold')
plt.colorbar(im8, ax=axes[2,1], fraction=0.046)

# 9. Critical Regions Overlay
axes[2,2].imshow(cmap_analysis, cmap='Greys', alpha=0.3, origin='lower')
critical_mask = cmap_analysis > 1.5
axes[2,2].contour(critical_mask, colors='red', linewidths=2, levels=[0.5])
axes[2,2].contourf(critical_mask, colors='red', alpha=0.3, levels=[0.5, 1])
axes[2,2].set_title('Critical Congestion Regions', fontsize=12, fontweight='bold')
axes[2,2].text(0.5, 0.95, f'{critical_mask.sum()}/{critical_mask.size} cells critical',
              transform=axes[2,2].transAxes, ha='center', va='top',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

for ax in axes.flatten():
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)

plt.tight_layout()
plt.savefig('detailed_congestion_analysis.png', dpi=200, bbox_inches='tight')
print("✅ Detailed analysis saved as 'detailed_congestion_analysis.png'")
plt.show()

# ============================================================
# STEP 6: CONGESTION STATISTICS
# ============================================================

print("\n" + "="*70)
print("📈 CONGESTION STATISTICS SUMMARY")
print("="*70)

stats_data = []
for scenario, cmap in zip(design_scenarios, congestion_maps):
    stats = {
        'Design': scenario['name'],
        'Mean': cmap.mean(),
        'Std': cmap.std(),
        'Min': cmap.min(),
        'Max': cmap.max(),
        'P95': np.percentile(cmap, 95),
        'Critical%': (cmap > 1.5).sum() / cmap.size * 100,
        'Warning%': ((cmap >= 0.8) & (cmap <= 1.5)).sum() / cmap.size * 100,
        'Safe%': (cmap < 0.8).sum() / cmap.size * 100
    }
    stats_data.append(stats)

stats_df = pd.DataFrame(stats_data)
print("\n" + stats_df.to_string(index=False))

# ============================================================
# STEP 7: EXPORT DATA
# ============================================================

print("\n" + "="*70)
print("💾 EXPORTING DATA")
print("="*70)

# Save congestion maps as numpy arrays
for i, (scenario, cmap) in enumerate(zip(design_scenarios, congestion_maps)):
    filename = f"congestion_map_{i+1}.npy"
    np.save(filename, cmap)
    print(f"✅ Saved: {filename}")

# Save statistics
stats_df.to_csv('congestion_map_statistics.csv', index=False)
print("✅ Statistics saved: congestion_map_statistics.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ SYNTHETIC MAP GENERATION COMPLETE")
print("="*70)

summary = f"""
📊 GENERATION SUMMARY:
   Maps generated:        {len(congestion_maps)}
   Grid size:             {generator.grid_size}x{generator.grid_size}
   Cells per map:         {generator.grid_size**2:,}

📈 OVERALL STATISTICS:
   Avg mean congestion:   {np.mean([c.mean() for c in congestion_maps]):.3f}
   Avg max congestion:    {np.mean([c.max() for c in congestion_maps]):.3f}
   Avg critical area:     {np.mean([((c > 1.5).sum() / c.size * 100) for c in congestion_maps]):.1f}%

💾 OUTPUT FILES:
   • synthetic_congestion_maps.png
   • detailed_congestion_analysis.png
   • congestion_map_*.npy (x{len(congestion_maps)})
   • congestion_map_statistics.csv

🎯 USE CASES:
   ✓ Design space exploration
   ✓ Routing algorithm testing
   ✓ Congestion mitigation validation
   ✓ EDA tool benchmarking
"""

print(summary)
print("="*70)
print("🎉 Ready for analysis and optimization!")
print("="*70)