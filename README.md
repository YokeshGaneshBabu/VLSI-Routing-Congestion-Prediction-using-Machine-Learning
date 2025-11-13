# 🚀 VLSI Routing Congestion Prediction using Machine Learning  
### **Realistic Dataset Generation + ML Pipeline + Visualization Suite**

This repository contains a **full end-to-end EDA + ML system** that:
- Generates **synthetic and realistic VLSI congestion datasets**  
- Trains **regression + classification models**  
- Evaluates on **unseen layouts**  
- Produces **professional congestion heatmaps & analysis visuals**  

This project attempts to model routing congestion similar to **physical design tools** (ICC2, Innovus, TritonRoute) using **ML + domain-aware synthetic data generation**.

---

# 🧩 **Project Features**

### ✅ 1. **Realistic Layout Generator**
Models true physical design effects:
- Macro placement with spacing rules  
- Cell clustering  
- Pin density modeling  
- Fanout (incl. H-tree clock nets)  
- Rent’s Rule–inspired wire length  
- Routing capacity by tech node  
- Macro blockage & proximity effects  
- DRC hotspot modeling  
- Thermal & IR-related power maps  

### ✅ 2. **ML Congestion Prediction**
- Random Forest Regression  
- Random Forest Classification  
- 13 input features (density, fanout, wirelength, macro presence, etc.)  
- 3-class congestion prediction (Green/Yellow/Red)

### ✅ 3. **Testing on Unseen Layouts**
- 5 new chip layouts  
- Performance: R², RMSE, MAE, MAPE  
- Per-layout metrics  
- Pixel-wise error heatmaps  

### ✅ 4. **Visualization Suite**
- Synthetic congestion maps  
- Realistic layouts  
- Detailed feature analysis  
- Error distribution  
- Confusion matrix  
- Prediction vs GT comparison  

---

# 🗂️ **Repository Structure**
<pre>
├── src/
│   ├── congestion_map_gen.py
│   ├── dataset_gen.py
│   ├── training.py
│   ├── testing.py
│
├── outputs/
│   ├── congestion_distribution.png
│   ├── detailed_congestion_analysis.png
│   ├── feature_importance.png
│   ├── prediction_quality.png
│   ├── realistic_layouts.png
│   ├── synthetic_congestion_maps.png
│   ├── test_analysis_metrics.png
│   ├── test_results_comparison.png
│   ├── congestion_map_statistics.csv
│   ├── test_results_summary.csv
└── README.md
<pre>

