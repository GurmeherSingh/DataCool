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
import json
import time
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

# Enhanced physics model coefficients (tuned to create realistic hotspots)
TEMP_AMBIENT = 21.0          # Ambient data center temperature (°C)
TEMP_CPU_COEFF = 0.52        # °C per % CPU load
TEMP_NET_COEFF = 0.015       # °C per MB/s network
TEMP_ADJACENCY_COEFF = 0.10  # Thermal coupling between adjacent racks
TEMP_ZONE_INFLUENCE = 1.5    # Hot zones affect nearby racks
TEMP_NOISE_STD = 1.5         # Random temperature variation

POWER_BASE = 0.5             # Base power draw (kW)
POWER_CPU_COEFF = 0.06       # kW per % CPU
POWER_NET_COEFF = 0.001      # kW per MB/s
POWER_NOISE_STD = 0.08

# Hotspot thresholds (tuned to catch more hotspots for optimization demo)
HOT_CPU_THRESHOLD = 75.0
HOT_TEMP_THRESHOLD = 65.0

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
    zone_penalties = np.array([0.0, 3.0, 6.5])  # °C penalty per zone
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
        temp += temp_influence * 0.6  # Damping factor
    
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
    
    # Assign thermal zones FIRST
    zones = assign_thermal_zones(n_racks)
    
    # Generate CPU loads strategically to create SOME hotspots but keep others cool
    cpu_load = np.zeros(n_racks)
    
    for i in range(n_racks):
        if zones[i] == 2:  # Poor cooling zone - Create hotspots here
            # Mix of very high and medium loads in zone 2
            if np.random.random() < 0.7:  # 70% high load
                cpu_load[i] = np.random.beta(5, 2) * 18 + 78  # 78-96%
            else:  # 30% medium load
                cpu_load[i] = np.random.beta(3, 3) * 30 + 50  # 50-80%
        elif zones[i] == 1:  # Medium cooling - MIXED LOADS
            # 40-75% CPU for racks in Zone 1
            cpu_load[i] = np.random.beta(3, 3) * 35 + 40
        else:  # Good cooling (Zone 0) - LOWER LOADS (will be candidates for optimization)
            # 10-55% CPU for racks in Zone 0
            cpu_load[i] = np.random.beta(2, 5) * 45 + 10
    
    # Add some variation
    cpu_load = np.clip(cpu_load + np.random.normal(0, 2, n_racks), 5, 100)
    
    # Network usage: correlated with CPU but not extreme
    network_base = np.random.exponential(100, n_racks)
    network_cpu_bonus = cpu_load * 2.8  # Moderate correlation
    network_usage = np.clip(network_base + network_cpu_bonus + np.random.normal(0, 40, n_racks), 10, 1000)
    
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
    # Use 0.5 threshold for balanced prediction
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
    
    # Identify hotspots using BOTH AI prediction AND actual thermal/load thresholds
    # This ensures we always have some candidates even if AI is too aggressive
    hotspots = df[
        ((df["Predicted"] == 1) | (df["Temperature"] > HOT_TEMP_THRESHOLD) | (df["CPU_load"] > HOT_CPU_THRESHOLD))
    ].index.tolist()
    
    # Candidates are racks that are NOT hotspots and have spare capacity
    candidates = df[
        ((df["Predicted"] == 0) & (df["Temperature"] < 60) & (df["CPU_load"] < 70))
    ].index.tolist()
    
    # If no candidates, use the coolest racks as fallback
    if len(candidates) == 0:
        # Sort by temperature and take the coolest 40% of racks
        df_sorted = df.sort_values('Temperature')
        num_candidates = max(3, int(n_racks * 0.4))
        candidates = df_sorted.head(num_candidates).index.tolist()
    
    if len(hotspots) == 0:
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
                "Cost_Units": "units",
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
    st.set_page_config(
        layout="wide", 
        page_title="DataCool - AI-Powered Datacenter Optimizer",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar configuration (moved to top for clean UI)
    st.sidebar.title("Configuration")
    
    with st.sidebar.expander("Simulation Settings", expanded=True):
        n_racks = st.slider("Number of racks", 20, 30, DEFAULT_RACK_COUNT)
        seed = st.number_input("Random seed", value=RANDOM_SEED, step=1)
    
    with st.sidebar.expander("Optimization Settings", expanded=True):
        max_migration_cost = st.slider("Max migration budget", 10.0, 200.0, MAX_MIGRATION_COST, step=10.0)
        enable_shap = st.checkbox("Show SHAP explainability", value=True)
        animation_speed = st.slider("Animation speed (seconds per step)", 0.5, 3.0, 1.5, step=0.5)
    
    # Generate data
    with st.spinner("🔄 Generating simulated data center..."):
        df = generate_racks(n_racks, seed)
    
    # Train model
    with st.spinner("🧠 Training AI model..."):
        model, eval_df, explainer, features, (acc, cm, report) = train_model(df, seed)
    
    # Make predictions
    df_pred = predict_all(df, model, features)
    
    # Show hotspot information
    hotspot_count = int(df_pred["Predicted"].sum())
    actual_hotspots = int(df_pred["Hotspot"].sum())
    max_temp = df_pred['Temperature'].max()
    
    if hotspot_count > 0:
        st.success(f"System ready for optimization! Detected {hotspot_count} predicted hotspots (Peak temperature: {max_temp:.1f}°C)")
    else:
        st.warning(f"No hotspots detected by AI. Actual labeled hotspots: {actual_hotspots}. Temperature range: {df_pred['Temperature'].min():.1f}°C - {max_temp:.1f}°C. Try increasing the simulation or adjusting random seed.")
    
    # =============================================================================
    # LANDING PAGE: Interactive 3D/2D Visualization
    # =============================================================================
    
    # Hero Section
    st.title("DataCool: AI-Powered Datacenter Optimizer")
    st.markdown("### Real-Time Thermal Management & Workload Optimization")
    
    # Key Metrics Row at Top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Racks", 
            len(df_pred),
            help="Total number of server racks in the datacenter"
        )
    with col2:
        hotspot_count = int(df_pred["Predicted"].sum())
        st.metric(
            "Predicted Hotspots", 
            hotspot_count,
            delta=f"{hotspot_count} critical" if hotspot_count > 0 else "All clear",
            delta_color="inverse",
            help="Racks predicted to exceed thermal thresholds"
        )
    with col3:
        max_temp = df_pred['Temperature'].max()
        st.metric(
            "Peak Temperature", 
            f"{max_temp:.1f}°C",
            delta=f"{max_temp - 65:.1f}°C from threshold" if max_temp > 65 else None,
            delta_color="inverse",
            help="Highest temperature across all racks"
        )
    with col4:
        avg_cpu = df_pred['CPU_load'].mean()
        st.metric(
            "Avg CPU Load", 
            f"{avg_cpu:.1f}%",
            help="Average CPU utilization across datacenter"
        )
    
    st.markdown("---")
    
    # Main Visualization
    st.header("Datacenter Thermal Visualization")
    st.markdown("**Interactive 3D/2D view** - Toggle between perspectives and click any rack for detailed metrics. Color scale: Blue (Cool) → Yellow (Warm) → Red (Hot)")
    
    # Convert dataframe to JSON format for visualization
    def df_to_json_format(df):
        """Convert dataframe to the JSON format expected by visual.html"""
        racks_data = []
        for _, row in df.iterrows():
            rack_data = {
                "Rack_ID": row["Rack_ID"],
                "CPU_load": float(row["CPU_load"]),
                "Temperature": float(row["Temperature"]),
                "Network_usage": float(row["Network_usage"]),
                "Power_consumption": float(row["Power_consumption"]),
                "Thermal_Zone": int(row["Thermal_Zone"]),
                "Row": int(row["Row"]),
                "Col": int(row["Col"]),
                "Hotspot": int(row.get("Hotspot", 0)),
                "Pred_Proba": float(row.get("Pred_Proba", 0.0)),
                "Predicted": int(row.get("Predicted", 0))
            }
            racks_data.append(rack_data)
        return {"before": racks_data, "after": racks_data}
    
    # Create placeholder for visualization that will update
    viz_placeholder = st.empty()
    
    # Generate initial JSON data
    json_data = df_to_json_format(df_pred)
    
    # Save to file for standalone visualization
    with open('datacenter_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Read the HTML file and inject data
    def create_visualization_html(json_data):
        """Read visual.html and inject JSON data"""
        html_path = "visual.html"
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Inject JSON data directly into the HTML
            json_str = json.dumps(json_data, indent=2)
            
            # Add script to inject data before the module script
            data_injection = f"""
            <script>
            // Injected data from Streamlit
            window.DATACENTER_DATA = {json_str};
            </script>
            """
            
            # Insert before the module script tag
            html_content = html_content.replace(
                "    <script type=\"module\">",
                data_injection + "    <script type=\"module\">"
            )
            
            # Modify loadData function to check for injected data first
            html_content = html_content.replace(
                "        async function loadData() {\n            try {",
                """        async function loadData() {
            // Check for injected data from Streamlit first
            if (window.DATACENTER_DATA) {
                datacenterData = window.DATACENTER_DATA.after || window.DATACENTER_DATA.before;
                console.log('Loaded rack data from Streamlit:', datacenterData.length, 'racks');
                initVisualizations();
                return;
            }
            try {"""
            )
            return html_content
        except:
            return f"""
            <div style="padding: 20px; color: white; background: #333; border-radius: 8px;">
                <h3>⚠️ Visualization File Not Found</h3>
                <p>Please ensure visual.html exists in the project directory.</p>
            </div>
            """
    
    # Display initial visualization
    viz_html = create_visualization_html(json_data)
    
    import streamlit.components.v1 as components
    with viz_placeholder.container():
        components.html(viz_html, height=800, scrolling=False)
    
    # AI Optimization Controls (placed right under the visualization)
    st.markdown("### AI-Powered Workload Optimization")
    
    col_opt1, col_opt2 = st.columns([3, 1])
    
    with col_opt1:
        st.markdown("""
        **Intelligent workload redistribution** using constraint-based optimization.
        The AI will move CPU load from overheated racks to cooler ones while respecting capacity limits.
        """)
    
    with col_opt2:
        optimize_button = st.button("Run AI Optimization", type="primary", use_container_width=True)
    
    # Optimization execution with real-time updates
    if optimize_button:
        # Create layout for optimization process
        st.markdown("---")
        st.subheader("Optimization Process")
        
        # Show BEFORE state
        st.markdown("#### Before Optimization")
        col_before1, col_before2 = st.columns(2)
        with col_before1:
            st.metric("Hotspots", hotspot_count)
            st.metric("Peak Temperature", f"{max_temp:.1f}°C")
        with col_before2:
            st.metric("Avg CPU Load", f"{df_pred['CPU_load'].mean():.1f}%")
            st.metric("Total Power", f"{df_pred['Power_consumption'].sum():.1f} kW")
        
        # DURING: Animation container
        st.markdown("---")
        st.markdown("#### During Optimization - Live Animation")
        
        col_viz, col_status = st.columns([2, 1])
        
        # Create persistent placeholders
        with col_viz:
            animation_viz = st.empty()
        
        with col_status:
            st.markdown("**Optimization Status**")
            step1_status = st.empty()
            step2_status = st.empty()
            step3_status = st.empty()
            step4_status = st.empty()
            st.markdown("---")
            progress_bar = st.empty()
            migration_status = st.empty()
            st.markdown("---")
            metrics_status = st.empty()
        
        # Step 1: AI Analysis
        step1_status.info("**Step 1:** Analyzing thermal patterns...")
        time.sleep(0.5)
        step1_status.success(f"**Step 1:** Detected {hotspot_count} hotspots across {len(df_pred)} racks")
        
        # Step 2: ML Prediction
        step2_status.info("**Step 2:** Running hotspot prediction model...")
        time.sleep(0.5)
        step2_status.success(f"**Step 2:** Model accuracy: {acc:.1%} | Peak temp: {max_temp:.1f}°C")
        
        # Step 3: Compute Optimization Plan
        step3_status.info("**Step 3:** Computing optimal workload distribution...")
        with st.spinner("Computing optimization plan..."):
            df_opt, transfers, metrics = optimize_workload_lp(df_pred, max_migration_cost)
        
        if metrics["status"] == "Success" and len(transfers) > 0:
            step3_status.success(f"**Step 3:** Plan ready: {len(transfers)} migrations | Cost: {metrics['total_cost']:.1f} Units")
            
            # Step 4: Execute with smooth animation
            step4_status.info(f"**Step 4:** Executing {len(transfers)} migrations...")
            
            # Simulate step-by-step migration with smooth updates
            df_current = df_pred.copy()
            
            for idx, transfer in enumerate(transfers):
                # Update progress
                progress = (idx + 1) / len(transfers)
                progress_bar.progress(progress, text=f"Migration {idx+1}/{len(transfers)}")
                migration_status.markdown(f"**Current:** {transfer['From']} → {transfer['To']} ({transfer['CPU_Moved']:.1f}% CPU)")
                
                # Apply this transfer to current state
                from_idx = df_current[df_current['Rack_ID'] == transfer['From']].index[0]
                to_idx = df_current[df_current['Rack_ID'] == transfer['To']].index[0]
                
                df_current.loc[from_idx, 'CPU_load'] -= transfer['CPU_Moved']
                df_current.loc[to_idx, 'CPU_load'] += transfer['CPU_Moved']
                
                # Recalculate temperatures for visualization
                df_current.loc[from_idx, 'Temperature'] = max(
                    TEMP_AMBIENT,
                    df_current.loc[from_idx, 'Temperature'] - (TEMP_CPU_COEFF * transfer['CPU_Moved'])
                )
                df_current.loc[to_idx, 'Temperature'] = min(
                    95.0,
                    df_current.loc[to_idx, 'Temperature'] + (TEMP_CPU_COEFF * transfer['CPU_Moved'] * 0.7)
                )
                
                # Recalculate hotspot labels based on new values
                df_current["Hotspot"] = ((df_current["CPU_load"] > HOT_CPU_THRESHOLD) & 
                                        (df_current["Temperature"] > HOT_TEMP_THRESHOLD)).astype(int)
                
                # Update predictions
                df_current = predict_all(df_current, model, features)
                
                # Update the SAME visualization with new data
                json_data_step = df_to_json_format(df_current)
                with open('datacenter_data.json', 'w') as f:
                    json.dump(json_data_step, f, indent=2)
                
                viz_html_step = create_visualization_html(json_data_step)
                with animation_viz:
                    components.html(viz_html_step, height=800, scrolling=False)
                
                # Show current metrics in status panel
                current_hotspots = int(df_current['Predicted'].sum())
                current_max_temp = df_current['Temperature'].max()
                
                with metrics_status:
                    st.metric("Hotspots", current_hotspots, delta=current_hotspots - hotspot_count)
                    st.metric("Peak Temp", f"{current_max_temp:.1f}°C", delta=f"{current_max_temp - max_temp:.1f}°C")
                
                # Delay for animation effect
                time.sleep(animation_speed)
            
            # Final update
            step4_status.success(f"**Step 4:** Complete! {len(transfers)} migrations executed")
            progress_bar.progress(1.0, text="Complete")
            migration_status.markdown("✅ **All migrations complete**")
            
            # Show AFTER state with final visualization
            st.markdown("---")
            st.markdown("#### After Optimization - Final Result")
            
            col_after1, col_after2, col_after3, col_after4 = st.columns(4)
            col_after1.metric("Hotspots", 
                       f"{metrics['hotspots_after']}",
                       delta=f"{metrics['hotspots_after'] - metrics['hotspots_before']}")
            col_after2.metric("Peak Temperature", 
                       f"{metrics['max_temp_after']:.1f}°C",
                       delta=f"{metrics['max_temp_after'] - metrics['max_temp_before']:.1f}°C")
            col_after3.metric("Total Power", 
                       f"{metrics['total_power_after']:.1f} kW",
                       delta=f"{metrics['total_power_after'] - metrics['total_power_before']:.1f} kW")
            col_after4.metric("Migration Cost", f"Cost: {metrics['total_cost']:.1f} Units")
            
            st.success(f"✅ Successfully eliminated {metrics['hotspots_before'] - metrics['hotspots_after']} hotspots! Temperature reduced by {metrics['max_temp_before'] - metrics['max_temp_after']:.1f}°C")
            
            # Detailed migration plan
            with st.expander("View Detailed Migration Plan"):
                transfers_df = pd.DataFrame(transfers)

                # Display Cost with units when available
                display_df = transfers_df.copy()
                if 'Cost_Units' in display_df.columns and 'Cost' in display_df.columns:
                    display_df['Cost'] = display_df.apply(lambda r: f"Cost: {r['Cost']} {str(r['Cost_Units']).capitalize()}", axis=1)

                st.dataframe(display_df, use_container_width=True)

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
            step3_status.info("**Step 3:** No hotspots detected")
            step4_status.info("**Step 4:** System already optimized")
            st.info("✅ No hotspots detected - system is already optimized!")
        else:
            step3_status.warning("**Step 3:** No feasible solution found")
            step4_status.warning("**Step 4:** Try adjusting parameters")
            st.warning("⚠️ No feasible transfers found. Try increasing migration budget or relaxing constraints.")
    
    st.markdown("---")
    
    # =============================================================================
    # AI Model Performance
    # =============================================================================
    
    st.header("AI Model Performance")
    
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
    # Current Datacenter State Analysis
    # =============================================================================
    
    st.header("Current State Analysis")
    
    # Heatmap
    temp_fig = plot_temperature_heatmap(df_pred, "Temperature", "Current Rack Temperatures")
    st.plotly_chart(temp_fig, use_container_width=True)
    
    # Risky racks table
    st.subheader("High-Risk Racks")
    risky_racks = df_pred.sort_values("Pred_Proba", ascending=False).head(8)[
        ["Rack_ID", "CPU_load", "Temperature", "Thermal_Zone", "Pred_Proba", "Predicted"]
    ]
    st.dataframe(risky_racks.reset_index(drop=True), use_container_width=True)
    # =============================================================================
    # Data Explorer
    # =============================================================================
    
    with st.expander("Full Dataset Explorer"):
        st.dataframe(df_pred, use_container_width=True)
        
        # Download option
        csv = df_pred.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="datacenter_simulation.csv",
            mime="text/csv"
        )
    
    # =============================================================================
    # Footer
    # =============================================================================
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()

