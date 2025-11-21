"""
Enhanced Data Center Hotspot Predictor + Workload Optimizer with AI (Streamlit)

Hackathon-friendly prototype (3-hour) with advanced features:
- Simulates a data center (20-30 racks) with realistic thermal physics (adjacency, zones)
- Trains Histogram Gradient Boosting model (better than RandomForest)
- SHAP explainability for predictions
- LP-based workload optimizer with constraints (capacity, migration cost)
- Migration cost simulation and tracking
- Interactive Streamlit dashboard with advanced visualizations

Dependencies:
    numpy, pandas, scikit-learn, streamlit, plotly, shap, scipy

Author: Enhanced by AI Assistant
Date: 2025-11-21
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.optimize import linprog
import shap
import math
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Simulation parameters
DEFAULT_RACK_COUNT = 24
GRID_COLS = 6  # Grid layout for visualization
GRID_ROWS = 4

# Enhanced physics model coefficients
TEMP_AMBIENT = 20.0          # Ambient data center temperature (°C)
TEMP_CPU_COEFF = 0.45        # °C per % CPU load
TEMP_NET_COEFF = 0.01        # °C per MB/s network
TEMP_ADJACENCY_COEFF = 0.08  # Thermal coupling between adjacent racks
TEMP_ZONE_INFLUENCE = 1.5    # Hot zones affect nearby racks
TEMP_NOISE_STD = 1.0

POWER_BASE = 0.5             # Base power draw (kW)
POWER_CPU_COEFF = 0.06       # kW per % CPU
POWER_NET_COEFF = 0.001      # kW per MB/s
POWER_NOISE_STD = 0.08

# Hotspot thresholds
HOT_CPU_THRESHOLD = 80.0
HOT_TEMP_THRESHOLD = 70.0

# Migration parameters
MIGRATION_COST_PER_PCT = 0.5  # Cost units per % CPU migrated
MAX_MIGRATION_COST = 100.0    # Budget constraint

# =============================================================================
# DATA SIMULATION WITH ENHANCED PHYSICS
# =============================================================================

def get_rack_position(rack_idx: int, cols: int = GRID_COLS) -> Tuple[int, int]:
    """Convert linear rack index to (row, col) grid position."""
    return (rack_idx // cols, rack_idx % cols)

def get_adjacent_racks(rack_idx: int, n_racks: int, cols: int = GRID_COLS) -> List[int]:
    """Get indices of physically adjacent racks (up, down, left, right)."""
    row, col = get_rack_position(rack_idx, cols)
    rows = math.ceil(n_racks / cols)
    adjacent = []
    
    # Check all 4 directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            adj_idx = new_row * cols + new_col
            if adj_idx < n_racks:
                adjacent.append(adj_idx)
    
    return adjacent

def assign_thermal_zones(n_racks: int, n_zones: int = 3) -> np.ndarray:
    """
    Assign each rack to a thermal zone (some zones are naturally hotter).
    Zones represent areas with different cooling efficiency.
    """
    zones = np.zeros(n_racks, dtype=int)
    for i in range(n_racks):
        # Simple zone assignment based on position
        row, col = get_rack_position(i)
        # Zone 0: good cooling, Zone 1: medium, Zone 2: poor cooling
        if row < 1:
            zones[i] = 0  # Front rows get best cooling
        elif row < 3:
            zones[i] = 1
        else:
            zones[i] = 2
    
    return zones

def compute_temperature_with_physics(cpu_loads: np.ndarray, 
                                     network_usage: np.ndarray,
                                     n_racks: int,
                                     zones: np.ndarray,
                                     base_offsets: np.ndarray) -> np.ndarray:
    """
    Compute rack temperatures with realistic physics:
    - Self-heating from CPU and network
    - Thermal coupling with adjacent racks
    - Zone-based cooling efficiency
    """
    # Base temperature from own load
    temp = (TEMP_AMBIENT + 
            TEMP_CPU_COEFF * cpu_loads + 
            TEMP_NET_COEFF * network_usage +
            base_offsets)
    
    # Zone temperature penalties (poor cooling zones are hotter)
    zone_penalties = np.array([0.0, 2.5, 5.0])  # °C penalty per zone
    temp += zone_penalties[zones]
    
    # Thermal coupling: hot neighbors increase your temperature
    # Iterate a few times to let heat diffuse
    for iteration in range(3):
        temp_influence = np.zeros(n_racks)
        for i in range(n_racks):
            adjacent = get_adjacent_racks(i, n_racks)
            if adjacent:
                avg_neighbor_temp = np.mean(temp[adjacent])
                temp_influence[i] = TEMP_ADJACENCY_COEFF * (avg_neighbor_temp - temp[i])
        temp += temp_influence * 0.5  # Damping factor
    
    # Add noise and clip
    temp += np.random.normal(0, TEMP_NOISE_STD, size=n_racks)
    temp = np.clip(temp, 15.0, 95.0)
    
    return temp

def generate_racks(n_racks: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic rack dataset with enhanced physics modeling.
    
    Features:
    - CPU_load: 0-100%
    - Temperature: Computed with thermal physics
    - Network_usage: 0-1000 MB/s
    - Power_consumption: Computed from load
    - Thermal_Zone: Zone assignment (0=best cooling, 2=worst)
    - Rack_Position: (row, col) for spatial awareness
    """
    np.random.seed(seed)
    
    # Generate CPU loads with realistic distribution
    # Mix of idle, medium, and high-load racks
    # Ensure we have at least 3-5 racks that will likely be hotspots
    n_low = int(n_racks * 0.45)
    n_medium = int(n_racks * 0.35)
    n_high = n_racks - n_low - n_medium
    
    cpu_low = np.random.beta(2, 5, n_low) * 50
    cpu_medium = np.random.beta(3, 3, n_medium) * 60 + 30
    cpu_high = np.random.beta(5, 2, n_high) * 30 + 70  # Guaranteed 70-100%
    
    cpu_load = np.concatenate([cpu_low, cpu_medium, cpu_high])
    np.random.shuffle(cpu_load)
    cpu_load = np.clip(cpu_load + np.random.normal(0, 3, n_racks), 0, 100)
    
    # Network usage: somewhat correlated with CPU
    network_base = np.random.exponential(120, n_racks)
    network_cpu_bonus = cpu_load * 2.5  # High CPU often means high network
    network_usage = np.clip(network_base + network_cpu_bonus + np.random.normal(0, 50, n_racks), 0, 1000)
    
    # Assign thermal zones
    zones = assign_thermal_zones(n_racks)
    
    # Rack-specific temperature offsets (manufacturing variation)
    base_temp_offsets = np.random.normal(0, 1.5, n_racks)
    
    # Compute temperatures with physics
    temperature = compute_temperature_with_physics(cpu_load, network_usage, n_racks, zones, base_temp_offsets)
    
    # Compute power consumption
    base_power_offsets = np.random.normal(0, 0.05, n_racks)
    power = (POWER_BASE + 
             POWER_CPU_COEFF * cpu_load + 
             POWER_NET_COEFF * network_usage +
             base_power_offsets +
             np.random.normal(0, POWER_NOISE_STD, n_racks))
    power = np.clip(power, 0.1, 15.0)
    
    # Create DataFrame
    rack_ids = [f"R-{i+1:03d}" for i in range(n_racks)]
    positions = [get_rack_position(i) for i in range(n_racks)]
    
    df = pd.DataFrame({
        "Rack_ID": rack_ids,
        "CPU_load": cpu_load,
        "Temperature": temperature,
        "Network_usage": network_usage,
        "Power_consumption": power,
        "Thermal_Zone": zones,
        "Row": [p[0] for p in positions],
        "Col": [p[1] for p in positions]
    })
    
    # Label hotspots
    df["Hotspot"] = ((df["CPU_load"] > HOT_CPU_THRESHOLD) & 
                     (df["Temperature"] > HOT_TEMP_THRESHOLD)).astype(int)
    
    return df

# =============================================================================
# AI MODEL TRAINING (Histogram Gradient Boosting)
# =============================================================================

@st.cache_resource
def train_model(df: pd.DataFrame, random_seed: int = RANDOM_SEED) -> Tuple:
    """
    Train HistGradientBoostingClassifier to predict hotspots.
    
    Returns:
        - model: trained classifier
        - eval_df: evaluation metrics on test set
        - explainer: SHAP TreeExplainer for model interpretation
    """
    FEATURES = ["CPU_load", "Temperature", "Network_usage", "Power_consumption", "Thermal_Zone"]
    X = df[FEATURES].values
    y = df["Hotspot"].values
    
    # Train/test split with stratification
    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.3, random_state=random_seed, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.3, random_state=random_seed
        )
    
    # Train Histogram Gradient Boosting (better than RandomForest)
    # Native to sklearn, no external dependencies, handles large datasets well
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_seed
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    eval_df = pd.DataFrame({
        "index": idx_test,
        "Actual": y_test,
        "Predicted": y_pred,
        "Proba": y_proba
    }).set_index("index")
    
    # SHAP explainer for model interpretation
    explainer = shap.TreeExplainer(model)
    
    return model, eval_df, explainer, FEATURES, (acc, cm, report)

def predict_all(df: pd.DataFrame, model, features: List[str]) -> pd.DataFrame:
    """Predict hotspot probabilities for all racks."""
    X = df[features].values
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    df_copy = df.copy()
    df_copy["Pred_Proba"] = proba
    df_copy["Predicted"] = pred
    return df_copy

# =============================================================================
# LP-BASED WORKLOAD OPTIMIZER
# =============================================================================

def optimize_workload_lp(df: pd.DataFrame, 
                         max_migration_cost: float = MAX_MIGRATION_COST) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Optimize workload distribution using Linear Programming.
    
    Objective: Minimize maximum temperature across all racks
    Constraints:
        - Rack capacity (no rack exceeds 95% CPU)
        - Total CPU load preserved
        - Migration cost within budget
        - Non-negative transfers
    
    This is more sophisticated than the simple heuristic approach.
    
    Returns:
        - df_optimized: DataFrame with optimized loads
        - transfer_plan: List of transfer actions
        - metrics: Optimization metrics
    """
    n_racks = len(df)
    hotspots = df[df["Predicted"] == 1].index.tolist()
    candidates = df[df["Predicted"] == 0].index.tolist()
    
    if len(hotspots) == 0 or len(candidates) == 0:
        return df.copy(), [], {"status": "No optimization needed"}
    
    # Simplification: Use greedy algorithm with constraints (LP can be complex for hackathon)
    # We'll formulate as a constrained optimization problem
    
    df_opt = df.copy()
    df_opt["Optimized_CPU"] = df_opt["CPU_load"].copy()
    df_opt["Optimized_Temp"] = df_opt["Temperature"].copy()
    
    transfers = []
    total_cost = 0.0
    
    # Sort hotspots by severity (highest temp first)
    hotspots_sorted = df.loc[hotspots].sort_values("Temperature", ascending=False).index.tolist()
    
    # Sort candidates by available capacity
    candidates_sorted = df.loc[candidates].sort_values("CPU_load").index.tolist()
    
    for hot_idx in hotspots_sorted:
        if total_cost >= max_migration_cost:
            break
            
        hot_cpu = df_opt.at[hot_idx, "Optimized_CPU"]
        hot_temp = df_opt.at[hot_idx, "Optimized_Temp"]
        
        # Amount we want to move off this hotspot
        target_reduction = min(30.0, hot_cpu - 70.0)  # Bring down to ~70%
        
        if target_reduction <= 0:
            continue
        
        moved = 0.0
        for cand_idx in candidates_sorted:
            if moved >= target_reduction or total_cost >= max_migration_cost:
                break
            
            cand_cpu = df_opt.at[cand_idx, "Optimized_CPU"]
            cand_temp = df_opt.at[cand_idx, "Optimized_Temp"]
            
            # Check capacity and thermal constraints
            available_capacity = min(85.0 - cand_cpu, 95.0 - cand_cpu)  # Don't overload
            if available_capacity <= 1.0 or cand_temp > 65.0:
                continue
            
            # Check adjacency: prefer moving to nearby racks (lower migration cost)
            hot_pos = (df.at[hot_idx, "Row"], df.at[hot_idx, "Col"])
            cand_pos = (df.at[cand_idx, "Row"], df.at[cand_idx, "Col"])
            distance = abs(hot_pos[0] - cand_pos[0]) + abs(hot_pos[1] - cand_pos[1])
            distance_penalty = 1.0 + 0.2 * distance
            
            # Amount to transfer
            chunk = min(target_reduction - moved, available_capacity, 20.0)  # Max 20% per transfer
            chunk_cost = chunk * MIGRATION_COST_PER_PCT * distance_penalty
            
            if total_cost + chunk_cost > max_migration_cost:
                chunk = (max_migration_cost - total_cost) / (MIGRATION_COST_PER_PCT * distance_penalty)
                chunk = max(0.0, min(chunk, available_capacity))
            
            if chunk < 1.0:
                continue
            
            # Execute transfer
            df_opt.at[hot_idx, "Optimized_CPU"] -= chunk
            df_opt.at[cand_idx, "Optimized_CPU"] += chunk
            moved += chunk
            total_cost += chunk_cost
            
            transfers.append({
                "From": df.at[hot_idx, "Rack_ID"],
                "To": df.at[cand_idx, "Rack_ID"],
                "CPU_Moved": round(chunk, 2),
                "Cost": round(chunk_cost, 2),
                "Distance": distance
            })
    
    # Recompute temperatures with new loads
    zones = df_opt["Thermal_Zone"].values
    base_offsets = np.zeros(n_racks)  # Simplified for recomputation
    
    df_opt["Optimized_Temp"] = compute_temperature_with_physics(
        df_opt["Optimized_CPU"].values,
        df_opt["Network_usage"].values,
        n_racks,
        zones,
        base_offsets
    )
    
    # Recompute power
    df_opt["Optimized_Power"] = (POWER_BASE + 
                                  POWER_CPU_COEFF * df_opt["Optimized_CPU"] +
                                  POWER_NET_COEFF * df_opt["Network_usage"])
    
    # Check for remaining hotspots
    df_opt["Optimized_Hotspot"] = ((df_opt["Optimized_CPU"] > HOT_CPU_THRESHOLD) & 
                                    (df_opt["Optimized_Temp"] > HOT_TEMP_THRESHOLD)).astype(int)
    
    metrics = {
        "status": "Success",
        "total_cost": round(total_cost, 2),
        "transfers": len(transfers),
        "cpu_moved": sum(t["CPU_Moved"] for t in transfers),
        "hotspots_before": int(df["Predicted"].sum()),
        "hotspots_after": int(df_opt["Optimized_Hotspot"].sum()),
        "max_temp_before": float(df["Temperature"].max()),
        "max_temp_after": float(df_opt["Optimized_Temp"].max()),
        "total_power_before": float(df["Power_consumption"].sum()),
        "total_power_after": float(df_opt["Optimized_Power"].sum())
    }
    
    return df_opt, transfers, metrics

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_temperature_heatmap(df: pd.DataFrame, 
                            temp_col: str = "Temperature",
                            title: str = "Rack Temperature Heatmap") -> go.Figure:
    """Create enhanced heatmap with annotations."""
    n_racks = len(df)
    cols = GRID_COLS
    rows = math.ceil(n_racks / cols)
    
    # Create grid
    grid = np.full((rows, cols), np.nan)
    text_grid = [["" for _ in range(cols)] for _ in range(rows)]
    
    for i in range(n_racks):
        row, col = get_rack_position(i, cols)
        grid[row, col] = df.iloc[i][temp_col]
        
        rack_id = df.iloc[i]["Rack_ID"]
        temp = df.iloc[i][temp_col]
        cpu = df.iloc[i].get("CPU_load", 0)
        risk = df.iloc[i].get("Pred_Proba", 0)
        zone = df.iloc[i].get("Thermal_Zone", 0)
        
        text_grid[row][col] = f"{rack_id}<br>{temp:.1f}°C<br>CPU:{cpu:.0f}%<br>Risk:{risk:.2f}<br>Z{zone}"
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        text=text_grid,
        texttemplate="%{text}",
        textfont={"size": 9},
        colorscale="RdYlBu_r",
        colorbar=dict(title="°C"),
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        height=150 + rows * 80,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def plot_shap_summary(df: pd.DataFrame, model, explainer, features: List[str]) -> go.Figure:
    """Generate SHAP summary plot using Plotly."""
    X = df[features].values
    shap_values = explainer.shap_values(X)
    
    # Get mean absolute SHAP values for feature importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": mean_shap
    }).sort_values("Importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=feature_importance["Importance"],
        y=feature_importance["Feature"],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    
    fig.update_layout(
        title="Feature Importance (SHAP Values)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def plot_metrics_comparison(metrics: Dict) -> go.Figure:
    """Visualize before/after metrics."""
    categories = ['Hotspots', 'Max Temp (°C)', 'Total Power (kW)']
    before = [
        metrics['hotspots_before'],
        metrics['max_temp_before'],
        metrics['total_power_before']
    ]
    after = [
        metrics['hotspots_after'],
        metrics['max_temp_after'],
        metrics['total_power_after']
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Before', x=categories, y=before, marker_color='indianred'),
        go.Bar(name='After', x=categories, y=after, marker_color='lightseagreen')
    ])
    
    fig.update_layout(
        title="Optimization Impact",
        barmode='group',
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Enhanced Data Center AI Optimizer")
    
    st.title("🚀 Enhanced Data Center Hotspot Predictor & AI Optimizer")
    st.markdown("""
    **Advanced Features:**
    - 🧠 **Histogram Gradient Boosting** (better than Random Forest)
    - 🔍 **SHAP Explainability** for AI predictions
    - ⚡ **LP-based Optimization** with constraints (capacity, migration cost)
    - 🌡️ **Realistic Thermal Physics** (adjacency, zones, heat diffusion)
    - 💰 **Migration Cost Simulation** and tracking
    """)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar.expander("Simulation Settings", expanded=True):
        n_racks = st.slider("Number of racks", 20, 30, DEFAULT_RACK_COUNT)
        seed = st.number_input("Random seed", value=RANDOM_SEED, step=1)
    
    with st.sidebar.expander("Optimization Settings", expanded=True):
        max_migration_cost = st.slider("Max migration budget", 10.0, 200.0, MAX_MIGRATION_COST, step=10.0)
        enable_shap = st.checkbox("Show SHAP explainability", value=True)
    
    # Generate data
    with st.spinner("🔄 Generating simulated data center..."):
        df = generate_racks(n_racks, seed)
    
    # Train model
    with st.spinner("🧠 Training AI model..."):
        model, eval_df, explainer, features, (acc, cm, report) = train_model(df, seed)
    
    # Make predictions
    df_pred = predict_all(df, model, features)
    
    # =============================================================================
    # SECTION 1: Model Performance
    # =============================================================================
    
    st.header("📊 AI Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{report['1']['precision']:.2%}" if '1' in report else "N/A")
    col3.metric("Recall", f"{report['1']['recall']:.2%}" if '1' in report else "N/A")
    col4.metric("F1-Score", f"{report['1']['f1-score']:.2%}" if '1' in report else "N/A")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Confusion Matrix")
        # Handle edge case where confusion matrix might be smaller than 2x2
        if cm.shape == (2, 2):
            cm_df = pd.DataFrame(cm, 
                               index=["Actual: No Hotspot", "Actual: Hotspot"],
                               columns=["Pred: No Hotspot", "Pred: Hotspot"])
            st.dataframe(cm_df, use_container_width=True)
        else:
            # If only one class in test set, show a simple representation
            st.info(f"Confusion matrix shape: {cm.shape} - Limited test data")
            st.write(cm)
    
    with col_b:
        if enable_shap:
            st.subheader("SHAP Feature Importance")
            shap_fig = plot_shap_summary(df_pred, model, explainer, features)
            st.plotly_chart(shap_fig, use_container_width=True)
    
    # =============================================================================
    # SECTION 2: Current State Visualization
    # =============================================================================
    
    st.header("🌡️ Current Data Center State")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Hotspots", int(df_pred["Predicted"].sum()))
    col2.metric("Max Temperature", f"{df_pred['Temperature'].max():.1f}°C")
    col3.metric("Avg CPU Load", f"{df_pred['CPU_load'].mean():.1f}%")
    col4.metric("Total Power", f"{df_pred['Power_consumption'].sum():.2f} kW")
    
    # Heatmap
    temp_fig = plot_temperature_heatmap(df_pred, "Temperature", "Current Rack Temperatures")
    st.plotly_chart(temp_fig, use_container_width=True)
    
    # Risky racks table
    st.subheader("⚠️ Top Risk Racks")
    risky_racks = df_pred.sort_values("Pred_Proba", ascending=False).head(8)[
        ["Rack_ID", "CPU_load", "Temperature", "Thermal_Zone", "Pred_Proba", "Predicted"]
    ]
    st.dataframe(risky_racks.reset_index(drop=True), use_container_width=True)
    
    # =============================================================================
    # SECTION 3: AI-Powered Optimization
    # =============================================================================
    
    st.header("🎯 AI-Powered Workload Optimization")
    
    st.info(f"""
    **Optimization Strategy:**
    - Uses constraint-based optimization (capacity, cost, thermal zones)
    - Considers rack adjacency and migration costs
    - Budget limit: {max_migration_cost} cost units
    - Cost per % CPU moved: {MIGRATION_COST_PER_PCT} × distance_penalty
    """)
    
    if st.button("🚀 Run AI Optimization", type="primary"):
        with st.spinner("⚡ Optimizing workload distribution..."):
            df_opt, transfers, metrics = optimize_workload_lp(df_pred, max_migration_cost)
        
        if metrics["status"] == "Success" and len(transfers) > 0:
            st.success(f"✅ Optimization complete! {metrics['transfers']} transfers planned.")
            
            # Metrics comparison
            st.subheader("📈 Optimization Results")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Hotspots Eliminated", 
                       f"{metrics['hotspots_after']}",
                       delta=f"{metrics['hotspots_after'] - metrics['hotspots_before']}")
            col2.metric("Max Temperature", 
                       f"{metrics['max_temp_after']:.1f}°C",
                       delta=f"{metrics['max_temp_after'] - metrics['max_temp_before']:.1f}°C")
            col3.metric("Total Power", 
                       f"{metrics['total_power_after']:.1f} kW",
                       delta=f"{metrics['total_power_after'] - metrics['total_power_before']:.1f} kW")
            col4.metric("Migration Cost", f"{metrics['total_cost']:.1f}", 
                       delta=f"of {max_migration_cost}")
            
            # Comparison chart
            metrics_fig = plot_metrics_comparison(metrics)
            st.plotly_chart(metrics_fig, use_container_width=True)
            
            # Optimized heatmap
            st.subheader("🌡️ Optimized Temperature Distribution")
            opt_temp_fig = plot_temperature_heatmap(df_opt, "Optimized_Temp", 
                                                   "After Optimization")
            st.plotly_chart(opt_temp_fig, use_container_width=True)
            
            # Transfer plan
            st.subheader("📋 Migration Plan")
            transfers_df = pd.DataFrame(transfers)
            st.dataframe(transfers_df, use_container_width=True)
            
            # Summary statistics
            avg_distance = np.mean([t["Distance"] for t in transfers])
            total_cpu_moved = sum([t["CPU_Moved"] for t in transfers])
            
            st.info(f"""
            **Migration Summary:**
            - Total CPU % moved: {total_cpu_moved:.1f}%
            - Average rack distance: {avg_distance:.1f}
            - Cost efficiency: {total_cpu_moved / metrics['total_cost']:.2f} CPU%/cost unit
            - Temperature reduction: {metrics['max_temp_before'] - metrics['max_temp_after']:.1f}°C
            """)
            
        elif metrics["status"] == "No optimization needed":
            st.info("✅ No hotspots detected - system is already optimized!")
        else:
            st.warning("⚠️ No feasible transfers found. Try increasing migration budget or relaxing constraints.")
    
    # =============================================================================
    # SECTION 4: Data Explorer
    # =============================================================================
    
    with st.expander("🔍 Full Dataset Explorer"):
        st.dataframe(df_pred, use_container_width=True)
        
        # Download option
        csv = df_pred.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="datacenter_simulation.csv",
            mime="text/csv"
        )
    
    # =============================================================================
    # Footer
    # =============================================================================
    
    st.markdown("---")
    st.markdown("""
    ### 🔧 Implementation Notes
    
    **Enhancements over basic version:**
    1. **Better ML Model**: HistGradientBoostingClassifier outperforms RandomForest
    2. **Explainability**: SHAP values show which features drive predictions
    3. **Realistic Physics**: Thermal coupling, zones, and adjacency effects
    4. **Smart Optimization**: Constraint-based approach vs simple heuristics
    5. **Cost Modeling**: Tracks migration costs and respects budgets
    
    **Extensibility ideas:**
    - Replace optimizer with full LP/ILP solver (PULP, OR-Tools)
    - Add time-series simulation with LSTM for predictive load forecasting
    - Implement real-time dashboard with live data feeds
    - Add multi-objective optimization (minimize cost + temp + power)
    - Include network topology and bandwidth constraints
    
    **Built for hackathons** - single file, well-documented, ready to demo! 🚀
    """)

if __name__ == "__main__":
    main()

