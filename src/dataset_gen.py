import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

"""
ENHANCED REALISTIC VLSI CONGESTION DATASET GENERATOR
Based on industry EDA standards and design rules
"""

class RealisticCongestionGenerator:
    """
    Generates industry-standard routing congestion datasets
    with proper design rule dependencies
    """

    def __init__(self, grid_size=64, num_layouts=50):
        """
        Args:
            grid_size: Grid resolution (64x64 = 4096 samples per layout)
            num_layouts: Number of different chip layouts to generate
        """
        self.N = grid_size
        self.num_layouts = num_layouts

        # Industry-standard design rules
        self.DESIGN_RULES = {
            'min_macro_spacing': 5,      # Minimum distance between macros
            'routing_layers': 6,          # Typical chip has 6-10 metal layers
            'tracks_per_layer': 10,       # Routing tracks per layer
            'wire_pitch_nm': 48,          # 7nm node wire pitch
            'via_resistance': 0.1,        # Via resistance penalty
            'clock_tree_overhead': 0.15,  # Clock uses 15% of routing
        }

    def generate_realistic_layout(self, seed=42, tech_node='7nm'):
        """
        Generate ONE realistic chip layout with industry design rules
        """
        np.random.seed(seed)

        # ============================================================
        # 1. TECHNOLOGY NODE PARAMETERS
        # ============================================================
        tech_params = self._get_tech_params(tech_node)

        # ============================================================
        # 2. MACRO PLACEMENT (with spacing rules)
        # ============================================================
        macros, macro_coords = self._place_macros_with_rules()

        # ============================================================
        # 3. CELL DENSITY (clustered, not uniform)
        # ============================================================
        density = self._generate_cell_density()

        # ============================================================
        # 4. PIN DENSITY (depends on cells + macros)
        # ============================================================
        # Industry rule: Macros have 10-50 pins, cells have 2-6 pins
        macro_pins = macros * np.random.uniform(10, 50, (self.N, self.N))
        cell_pins = density * np.random.uniform(2, 6, (self.N, self.N))
        pin_density = macro_pins + cell_pins

        # ============================================================
        # 5. FANOUT (realistic decimal values)
        # ============================================================
        # Industry: Average fanout = 3-8 for standard cells
        # Clock nets: fanout 100-1000+
        # Power nets: fanout 1000+

        base_fanout = density * np.random.uniform(2.5, 8.5, (self.N, self.N))

        # Add high-fanout clock tree regions (10% of area)
        clock_regions = self._generate_clock_tree_regions()
        clock_fanout = clock_regions * np.random.uniform(50, 200, (self.N, self.N))

        fanout = base_fanout + clock_fanout

        # ============================================================
        # 6. WIRE LENGTH DISTRIBUTION
        # ============================================================
        # Industry: Most wires are short, few are global
        # Rent's rule: Wire length follows power-law distribution
        wire_length = self._generate_wire_lengths(density, macro_coords)

        # ============================================================
        # 7. ROUTING CAPACITY (layer-dependent)
        # ============================================================
        base_capacity = tech_params['base_capacity']

        # Capacity reduction rules:
        # 1. Macros block routing (80% reduction)
        # 2. High density reduces capacity (30% reduction)
        # 3. Design rule spacing reduces effective capacity

        capacity = np.full((self.N, self.N), base_capacity)
        capacity -= macros * (base_capacity * 0.8)  # Macro blockage
        capacity -= (density > 0.7) * (base_capacity * 0.3)  # High density penalty
        capacity = np.clip(capacity, base_capacity * 0.1, None)

        # ============================================================
        # 8. MACRO PROXIMITY EFFECT
        # ============================================================
        # Industry: Routing near macros has 2-3× higher congestion
        macro_prox = self._compute_macro_proximity(macro_coords)

        # ============================================================
        # 9. ROUTING DEMAND (complex interdependencies)
        # ============================================================
        # Demand formula based on Rent's rule and industry models:
        # Demand = f(density, pins, fanout, wire_length, clock_overhead)

        # Base demand from cell connections
        base_demand = (
            0.3 * density +           # More cells = more routing
            0.2 * pin_density / 10 +  # More pins = more nets
            0.15 * (fanout / 10) +    # High fanout = more wires
            0.1 * (wire_length / 100) # Long wires = more demand
        )

        # Clock tree overhead (15% additional demand)
        clock_demand = clock_regions * self.DESIGN_RULES['clock_tree_overhead']

        # Macro proximity increases demand (congestion walls)
        macro_effect = macro_prox * 0.25

        total_demand = base_demand + clock_demand + macro_effect

        # ============================================================
        # 10. DESIGN RULE CHECK (DRC) VIOLATIONS
        # ============================================================
        # Areas with spacing violations increase demand
        drc_violations = self._detect_drc_hotspots(density, macros)
        total_demand += drc_violations * 0.2

        # ============================================================
        # 11. FINAL CONGESTION CALCULATION
        # ============================================================
        # Industry formula: Congestion = Demand / Supply
        # With non-linear overflow penalty

        congestion = total_demand / capacity

        # Apply overflow penalty (>100% utilization is exponentially worse)
        overflow_mask = congestion > 1.0
        congestion[overflow_mask] = 1.0 + (congestion[overflow_mask] - 1.0) ** 1.5

        congestion = congestion.clip(0, 5)

        # Apply spatial smoothing (routing spreads to nearby tracks)
        congestion = gaussian_filter(congestion, sigma=1.0)

        # ============================================================
        # 12. ADDITIONAL REALISTIC FEATURES
        # ============================================================

        # Net criticality (timing-critical nets need better routing)
        criticality = self._compute_net_criticality(density, clock_regions)

        # Power grid density (IR drop affects routing)
        power_density = self._generate_power_grid(macros, density)

        # Temperature hotspots (high activity = routing challenges)
        temperature = self._compute_thermal_map(density, macros, fanout)

        # ============================================================
        # 13. CREATE DATAFRAME WITH ALL FEATURES
        # ============================================================

        df = pd.DataFrame({
            # Basic features (original)
            'density': density.flatten(),
            'pin_density': pin_density.flatten(),
            'fanout': fanout.flatten(),
            'macro': macros.flatten(),
            'capacity': capacity.flatten(),
            'macro_proximity': macro_prox.flatten(),

            # NEW: Advanced features
            'wire_length': wire_length.flatten(),
            'clock_region': clock_regions.flatten(),
            'drc_violations': drc_violations.flatten(),
            'net_criticality': criticality.flatten(),
            'power_density': power_density.flatten(),
            'temperature': temperature.flatten(),

            # Target
            'congestion': congestion.flatten()
        })

        # Add metadata
        df['tech_node'] = tech_node
        df['layout_id'] = seed

        return df, congestion

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _get_tech_params(self, tech_node):
        """Technology node specific parameters"""
        params = {
            '180nm': {'base_capacity': 2.0, 'wire_resistance': 0.1},
            '90nm':  {'base_capacity': 3.0, 'wire_resistance': 0.15},
            '45nm':  {'base_capacity': 4.0, 'wire_resistance': 0.2},
            '28nm':  {'base_capacity': 5.0, 'wire_resistance': 0.25},
            '14nm':  {'base_capacity': 6.0, 'wire_resistance': 0.3},
            '7nm':   {'base_capacity': 8.0, 'wire_resistance': 0.35},
            '5nm':   {'base_capacity': 10.0, 'wire_resistance': 0.4},
        }
        return params.get(tech_node, params['7nm'])

    def _place_macros_with_rules(self):
        """Place macros with minimum spacing rules"""
        macros = np.zeros((self.N, self.N))
        macro_coords = []

        num_macros = np.random.randint(4, 10)  # 4-10 macros per chip
        min_spacing = self.DESIGN_RULES['min_macro_spacing']

        for _ in range(num_macros):
            placed = False
            attempts = 0

            while not placed and attempts < 50:
                cx = np.random.randint(8, self.N-8)
                cy = np.random.randint(8, self.N-8)

                # Check spacing rule
                valid = True
                for prev_cx, prev_cy in macro_coords:
                    dist = np.sqrt((cx-prev_cx)**2 + (cy-prev_cy)**2)
                    if dist < min_spacing:
                        valid = False
                        break

                if valid:
                    size = np.random.randint(3, 7)  # Variable macro sizes
                    macros[cx-size:cx+size, cy-size:cy+size] = 1
                    macro_coords.append((cx, cy))
                    placed = True

                attempts += 1

        return macros, macro_coords

    def _generate_cell_density(self):
        """Generate realistic clustered cell density"""
        density = np.zeros((self.N, self.N))

        # Create 15-25 density clusters (functional blocks)
        num_clusters = np.random.randint(15, 25)

        for _ in range(num_clusters):
            cx, cy = np.random.randint(0, self.N), np.random.randint(0, self.N)
            x = np.arange(self.N)
            y = np.arange(self.N)
            xv, yv = np.meshgrid(x, y)

            # Variable cluster sizes (some tight, some spread)
            spread = np.random.uniform(50, 150)
            intensity = np.random.uniform(0.5, 1.5)

            blob = intensity * np.exp(-((xv-cx)**2 + (yv-cy)**2) / spread)
            density += blob

        density = density / density.max()
        density = np.clip(density, 0, 1)

        return density

    def _generate_clock_tree_regions(self):
        """Generate clock tree distribution (H-tree or spine)"""
        clock = np.zeros((self.N, self.N))

        # H-tree pattern (industry standard)
        mid = self.N // 2

        # Main spine (vertical + horizontal)
        clock[mid-2:mid+2, :] = 1
        clock[:, mid-2:mid+2] = 1

        # Secondary branches
        clock[mid//2-1:mid//2+1, :] = 0.5
        clock[3*mid//2-1:3*mid//2+1, :] = 0.5

        return clock

    def _generate_wire_lengths(self, density, macro_coords):
        """Generate wire length distribution using Rent's rule"""
        wire_length = np.zeros((self.N, self.N))

        # Short local wires (most common)
        local_wires = density * np.random.uniform(10, 50, (self.N, self.N))

        # Medium semi-global wires
        for cx, cy in macro_coords:
            x = np.arange(self.N)
            y = np.arange(self.N)
            xv, yv = np.meshgrid(x, y)
            dist = np.sqrt((xv-cx)**2 + (yv-cy)**2)
            wire_length += dist * 2  # Macros generate global wires

        wire_length += local_wires
        wire_length = np.clip(wire_length, 10, 500)

        return wire_length

    def _compute_macro_proximity(self, macro_coords):
        """Compute distance-based macro influence"""
        macro_prox = np.zeros((self.N, self.N))

        for x in range(self.N):
            for y in range(self.N):
                if len(macro_coords) > 0:
                    distances = [np.sqrt((x-cx)**2 + (y-cy)**2) for cx, cy in macro_coords]
                    min_dist = min(distances)
                    macro_prox[x, y] = 1 / (min_dist + 1)

        return macro_prox

    def _detect_drc_hotspots(self, density, macros):
        """Detect design rule violation hotspots"""
        # High density + macro proximity = DRC issues
        drc = np.zeros((self.N, self.N))

        # Convolution to detect crowded regions
        from scipy.signal import convolve2d
        kernel = np.ones((5, 5)) / 25
        crowding = convolve2d(density, kernel, mode='same')

        drc = (crowding > 0.6).astype(float) * 0.5
        drc += (macros > 0).astype(float) * 0.3

        return drc

    def _compute_net_criticality(self, density, clock_regions):
        """Compute timing-critical net distribution"""
        # Critical paths: high density areas + clock regions
        criticality = density * 0.5 + clock_regions * 0.8
        criticality = np.clip(criticality, 0, 1)
        return criticality

    def _generate_power_grid(self, macros, density):
        """Generate power grid density"""
        # Macros and high-density areas need more power
        power = macros * 0.6 + density * 0.4
        power = gaussian_filter(power, sigma=2.0)
        return power

    def _compute_thermal_map(self, density, macros, fanout):
        """Compute thermal hotspots"""
        # High switching activity = heat = routing challenges
        temperature = (
            density * 0.4 +           # More cells = more heat
            macros * 0.3 +            # Macros generate heat
            (fanout / fanout.max()) * 0.3  # High fanout = activity
        )
        temperature = gaussian_filter(temperature, sigma=3.0)
        return temperature

    # ============================================================
    # MAIN GENERATION FUNCTION
    # ============================================================

    def generate_large_dataset(self, tech_nodes=['7nm', '14nm', '28nm']):
        """
        Generate large-scale realistic dataset

        Returns:
            DataFrame with 10K+ samples from multiple layouts
        """
        all_data = []
        all_congestion_maps = []

        print(f"🔧 Generating {self.num_layouts} chip layouts...")
        print(f"   Grid size: {self.N}×{self.N} = {self.N**2} samples per layout")
        print(f"   Total samples: {self.num_layouts * self.N**2:,}")
        print()

        for i in tqdm(range(self.num_layouts), desc="Generating layouts"):
            tech_node = np.random.choice(tech_nodes)
            df, cong_map = self.generate_realistic_layout(
                seed=42 + i,
                tech_node=tech_node
            )
            all_data.append(df)
            all_congestion_maps.append(cong_map)

        # Combine all layouts
        final_df = pd.concat(all_data, ignore_index=True)

        print(f"\n✅ Dataset generation complete!")
        print(f"   Total samples: {len(final_df):,}")
        print(f"   Features: {len(final_df.columns) - 3}")  # Exclude target, tech, layout_id
        print(f"   Layouts: {self.num_layouts}")

        return final_df, all_congestion_maps


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Create generator
    generator = RealisticCongestionGenerator(
        grid_size=64,      # 64×64 = 4096 samples per layout
        num_layouts=10     # 10 layouts = 40,960 total samples
    )

    # Generate dataset
    df, congestion_maps = generator.generate_large_dataset(
        tech_nodes=['7nm', '14nm', '28nm']
    )

    # Show statistics
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    print(f"\nShape: {df.shape}")
    print(f"\nFeatures:\n{df.columns.tolist()}")
    print(f"\nCongestion statistics:")
    print(df['congestion'].describe())

    print(f"\nTechnology node distribution:")
    print(df['tech_node'].value_counts())

    print(f"\nSample data:")
    print(df.head())

    # Visualize first 5 layouts
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i in range(min(5, len(congestion_maps))):
        im = axes[i].imshow(congestion_maps[i], cmap='RdYlGn_r', vmin=0, vmax=3)
        axes[i].set_title(f'Layout {i+1}', fontsize=12, fontweight='bold')
        axes[i].axis('off')

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Congestion')
    plt.suptitle('First 5 Generated Chip Layouts (Enhanced Realism)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('realistic_layouts.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved as 'realistic_layouts.png'")
    plt.show()

    # Save dataset
    df.to_csv('realistic_congestion_dataset.csv', index=False)
    print(f"\n💾 Dataset saved as 'realistic_congestion_dataset.csv'")
