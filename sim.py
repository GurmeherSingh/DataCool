# datacenter_ai_sim.py
"""
Data Center Hotspot Predictor + Workload Optimizer (Streamlit)

Single-file hackathon prototype (3-hour friendly) that:
- Simulates a small data center (20-30 racks) with correlated features.
- Trains an AI model (Random Forest) to predict "Hotspot" racks.
- Produces prediction probabilities.
- Implements a simple, explainable recommendation engine to redistribute workload.
- Displays an interactive Streamlit dashboard with:
    * Heatmap of rack temperatures (grid layout),
    * Sliders to tweak CPU load per rack,
    * Dynamic AI predictions,
    * "Apply AI Recommendation" button that simulates the effect of workload redistribution
      and shows projected improvements.

Dependencies:
    numpy, pandas, scikit-learn, streamlit, plotly

Author: ChatGPT (hackathon-friendly)
Date: 2025-11-21
"""

from typing import Tuple, List, Dict
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import math
import random

# --------------------
# Utility / Config
# --------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# We choose a default number of racks between 20 and 30 as required.
DEFAULT_RACK_COUNT = 24  # flexible — user can change
GRID_COLS = 6  # how many columns to lay out racks in the heatmap grid

# These coefficients govern the synthetic physics of rack temperature and power:
TEMP_BASE = 20.0             # base temperature (°C) when CPU_load is 0 and network is minimal
TEMP_CPU_COEFF = 0.45        # °C per percentage-point of CPU load
TEMP_NET_COEFF = 0.01        # °C per MB/s of network usage (small effect)
TEMP_NOISE_STD = 1.2         # random noise standard deviation for temperature

POWER_BASE = 0.5             # kW base draw even at idle
POWER_CPU_COEFF = 0.05       # kW per percentage-point of CPU load
POWER_NET_COEFF = 0.0008     # kW per MB/s of network (small)
POWER_NOISE_STD = 0.05       # noise for power readings

# Hotspot rule (label): CPU_load > 80% and Temperature > 70°C -> Hotspot
HOT_CPU_THRESHOLD = 80.0
HOT_TEMP_THRESHOLD = 70.0

# Slab of load to move off hotspots in recommendations (percent of CPU load)
TRANSFER_PERCENT = 0.20  # 20% of the hotspot's CPU load will be redistributed (configurable)

# --------------------
# Data Simulation
# --------------------

def generate_racks(n_racks: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic dataset of racks with correlated features.

    Each rack row includes:
        - Rack_ID (string)
        - CPU_load (0-100%)
        - Temperature (°C) computed from CPU_load and network usage + noise
        - Network_usage (MB/s) (0-1000)
        - Power_consumption (kW) computed from CPU_load + network usage + noise
        - Hotspot label (1/0)

    The simulation enforces correlation:
        higher CPU_load -> higher temperature and higher power consumption
    """
    np.random.seed(seed)
    n = n_racks

    # We'll introduce per-rack base offsets to add heterogeneity (some racks are 'warmer' by design)
    rack_offsets_temp = np.random.normal(loc=0.0, scale=1.5, size=n)
    rack_offsets_power = np.random.normal(loc=0.0, scale=0.05, size=n)

    # CPU load distribution: skewed towards moderate loads; some high-load racks
    # Mix of distributions to create interesting cases
    cpu_load_1 = np.random.beta(a=2, b=5, size=int(n * 0.6)) * 60                # many moderate loads
    cpu_load_2 = (np.random.beta(a=5, b=2, size=n - int(n * 0.6)) * 40) + 40    # some high loads
    cpu_load = np.concatenate([cpu_load_1, cpu_load_2])
    np.random.shuffle(cpu_load)
    cpu_load = np.clip(cpu_load + np.random.normal(0, 3, size=n), 0, 100)  # slight noise

    # Network usage: broad distribution 0-1000 MB/s
    network_usage = np.clip(np.random.exponential(scale=150.0, size=n), 0, 1000)

    # Temperature model (synthetic physics-inspired):
    temp = (TEMP_BASE
            + TEMP_CPU_COEFF * cpu_load
            + TEMP_NET_COEFF * network_usage
            + rack_offsets_temp
            + np.random.normal(0, TEMP_NOISE_STD, size=n))

    # Power consumption model:
    power = (POWER_BASE
             + POWER_CPU_COEFF * cpu_load
             + POWER_NET_COEFF * network_usage
             + rack_offsets_power
             + np.random.normal(0, POWER_NOISE_STD, size=n))

    # Clip to realistic bounds
    temp = np.clip(temp, 15.0, 95.0)
    power = np.clip(power, 0.05, 15.0)

    # Rack IDs
    rack_ids = [f"R-{i+1:03d}" for i in range(n)]

    df = pd.DataFrame({
        "Rack_ID": rack_ids,
        "CPU_load": cpu_load,
        "Temperature": temp,
        "Network_usage": network_usage,
        "Power_consumption": power
    })

    # Label hotspots per provided rule
    df["Hotspot"] = ((df["CPU_load"] > HOT_CPU_THRESHOLD) & (df["Temperature"] > HOT_TEMP_THRESHOLD)).astype(int)

    return df

# --------------------
# AI Model Training & Prediction
# --------------------

@st.cache_data(ttl=3600)
def train_model(df: pd.DataFrame, random_seed: int = RANDOM_SEED) -> Tuple[RandomForestClassifier, StandardScaler, pd.DataFrame]:
    """
    Train a RandomForest classifier to predict Hotspot.

    Returns:
        - trained model,
        - scaler used for features,
        - evaluation DataFrame containing actual/predicted/proba for the test set
    """
    # Features chosen
    FEATURES = ["CPU_load", "Temperature", "Network_usage", "Power_consumption"]
    X = df[FEATURES].values
    y = df["Hotspot"].values

    # Small dataset -> use stratify if possible
    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.3, random_state=random_seed, stratify=y)
    except ValueError:
        # fallback if stratify fails (e.g. no hotspots in small sample)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.3, random_state=random_seed)

    # Standardize numeric features for model stability (though tree-based models don't need it, it's instructive)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model: Random Forest (robust, fast to train)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_seed)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1 (Hotspot)

    eval_df = pd.DataFrame({
        "index": idx_test,
        "Actual": y_test,
        "Predicted": y_pred,
        "Proba": y_proba
    }).set_index("index")

    return model, scaler, eval_df

def predict_all(df: pd.DataFrame, model: RandomForestClassifier, scaler: StandardScaler) -> pd.DataFrame:
    """
    Predict hotspot probabilities for all racks in df, returning df copy with added columns:
        - Predicted (0/1)
        - Pred_Proba (float)
    """
    FEATURES = ["CPU_load", "Temperature", "Network_usage", "Power_consumption"]
    X = df[FEATURES].values
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)

    df_copy = df.copy()
    df_copy["Pred_Proba"] = proba
    df_copy["Predicted"] = pred
    return df_copy

# --------------------
# Recommendation Engine
# --------------------

def recommend_redistribution(df: pd.DataFrame,
                             transfer_fraction: float = TRANSFER_PERCENT,
                             cpu_target_threshold: float = 60.0,
                             temp_target_threshold: float = 60.0) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Given a DataFrame with current rack states and predictions, generate a redistribution plan:

    Strategy (simple, explainable):
      1. Identify racks predicted as hotspots (Predicted == 1) or actual hotspot label==1.
      2. For each hotspot, propose transferring `transfer_fraction` of its current CPU_load
         to one or multiple candidate racks that:
            - have CPU_load < cpu_target_threshold AND Temperature < temp_target_threshold
            - sorted by lowest CPU_load first (to receive load)
      3. The transferred CPU reduces source's CPU_load and increases target's CPU_load.
      4. Recompute temp and power using the same synthetic models (linear approx) to project improvements.

    Returns:
      - new_df: DataFrame with projected CPU_load, Temperature, Power_consumption after redistribution
      - plan: list of transfer actions for display
    """
    df = df.copy().reset_index(drop=True)

    # Ensure the necessary columns exist
    if "Predicted" not in df.columns:
        raise ValueError("DataFrame must include 'Predicted' column (0/1) before calling recommendation engine.")

    plan = []  # human-readable actions
    df["Projected_CPU"] = df["CPU_load"].astype(float)
    df["Projected_Temperature"] = df["Temperature"].astype(float)
    df["Projected_Power"] = df["Power_consumption"].astype(float)

    # Candidate pools (indices in df)
    receivers = set(df[(df["CPU_load"] < cpu_target_threshold) & (df["Temperature"] < temp_target_threshold)].index.tolist())

    # We'll iterate hotspots in descending risk order (by Pred_Proba), to prioritize most risky racks
    hotspots = df[df["Predicted"] == 1].sort_values("Pred_Proba", ascending=False)

    for src_idx, src_row in hotspots.iterrows():
        src_id = src_row["Rack_ID"]
        src_cpu = df.at[src_idx, "Projected_CPU"]
        amount_to_move = src_cpu * transfer_fraction  # absolute CPU percentage points to move

        # If no receivers available or amount is trivial, continue
        if len(receivers) == 0 or amount_to_move < 1.0:
            continue

        # Find best receiver candidates by current projected CPU (lowest first)
        receiver_candidates = sorted(list(receivers), key=lambda i: df.at[i, "Projected_CPU"])

        moved = 0.0
        # We'll try to distribute the amount across multiple receivers fairly, but stop when amount is moved
        for r_idx in receiver_candidates:
            if moved >= amount_to_move:
                break
            # How much can this receiver accept without exceeding 90% CPU (safety cap)
            receiver_cpu = df.at[r_idx, "Projected_CPU"]
            capacity = max(0.0, 90.0 - receiver_cpu)  # want to avoid creating new hotspots
            if capacity <= 0.5:
                # receiver nearly full — remove from candidate pool
                receivers.discard(r_idx)
                continue

            # allocate a chunk proportional to available capacity (simple heuristic)
            chunk = min(amount_to_move - moved, capacity, amount_to_move * 0.6)  # avoid giving everything to one rack
            if chunk < 0.5:
                continue  # chunk too small to be meaningful

            # Apply transfer
            df.at[src_idx, "Projected_CPU"] -= chunk
            df.at[r_idx, "Projected_CPU"] += chunk
            moved += chunk

            # Record action
            plan.append({
                "from": src_id,
                "to": df.at[r_idx, "Rack_ID"],
                "cpu_moved_pct": round(chunk, 2)
            })

            # If receiver is now too loaded, remove from receivers
            if df.at[r_idx, "Projected_CPU"] >= 85.0:
                receivers.discard(r_idx)

        # If we couldn't move enough, we moved what we could; move on to next hotspot

    # After transfers, recompute projected temperature and power using the same linear models as in simulation
    # Reuse our synthetic physics constants (linear approximations)
    df["Projected_Temperature"] = (TEMP_BASE
                                  + TEMP_CPU_COEFF * df["Projected_CPU"]
                                  + TEMP_NET_COEFF * df["Network_usage"]
                                  + np.random.normal(0, TEMP_NOISE_STD, size=len(df)))
    df["Projected_Temperature"] = np.clip(df["Projected_Temperature"], 15.0, 95.0)

    df["Projected_Power"] = (POWER_BASE
                             + POWER_CPU_COEFF * df["Projected_CPU"]
                             + POWER_NET_COEFF * df["Network_usage"]
                             + np.random.normal(0, POWER_NOISE_STD, size=len(df)))
    df["Projected_Power"] = np.clip(df["Projected_Power"], 0.05, 15.0)

    # Re-evaluate hotspot status after projection
    df["Projected_Hotspot"] = ((df["Projected_CPU"] > HOT_CPU_THRESHOLD) & (df["Projected_Temperature"] > HOT_TEMP_THRESHOLD)).astype(int)

    return df, plan

# --------------------
# Visualization Helpers (Plotly)
# --------------------

def make_rack_grid(df: pd.DataFrame, value_col: str = "Temperature", cols: int = GRID_COLS) -> Tuple[np.ndarray, List[str]]:
    """
    Convert df into a 2D grid for plotting with a heatmap.
    value_col indicates which numeric column to visualize (e.g., Temperature, Projected_Temperature)

    Returns:
        - grid: 2D numpy array (rows x cols) with value_col or np.nan where grid is empty
        - labels: list of Rack_IDs laid out row-major by grid; used for annotations
    """
    n = len(df)
    cols = cols
    rows = math.ceil(n / cols)
    arr = np.full((rows, cols), np.nan)
    labels = []
    for i in range(n):
        r = i // cols
        c = i % cols
        arr[r, c] = df.iloc[i][value_col]
        labels.append(df.iloc[i]["Rack_ID"])
    return arr, labels

def plot_temperature_heatmap(df: pd.DataFrame, value_col="Temperature", risk_col="Pred_Proba", cols: int = GRID_COLS, title: str = "Rack Temperatures"):
    """
    Create a Plotly heatmap of the rack grid with colors representing value_col
    and text annotations showing Rack_ID and risk (rounded probability).
    """
    grid_vals, labels = make_rack_grid(df, value_col=value_col, cols=cols)
    rows, cols = grid_vals.shape

    # Build annotation text matrix
    text = []
    for r in range(rows):
        row_text = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(df):
                rid = df.iloc[idx]["Rack_ID"]
                val = grid_vals[r, c]
                risk = df.iloc[idx][risk_col] if risk_col in df.columns else 0.0
                row_text.append(f"{rid}<br>{val:.1f}°C<br>Risk {risk:.2f}")
            else:
                row_text.append("")  # empty cell
        text.append(row_text)

    # Heatmap trace
    heatmap = go.Heatmap(
        z=grid_vals,
        text=text,
        hoverinfo='text',
        colorscale='YlOrRd',
        colorbar=dict(title="°C"),
        zmin=np.nanmin(grid_vals),
        zmax=np.nanmax(grid_vals)
    )
    layout = go.Layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20),
        height=300 + rows * 60
    )
    fig = go.Figure(data=[heatmap], layout=layout)
    # Add annotations manually so rack ids and risk show
    annotations = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < len(df):
                annotations.append(
                    dict(
                        x=c,
                        y=r,
                        text=text[r][c],
                        showarrow=False,
                        font=dict(color='black', size=10)
                    )
                )
    fig.update_layout(annotations=annotations)
    # Flip y axis to have first row at the top like a grid
    fig.update_yaxes(autorange='reversed')
    return fig

# --------------------
# Streamlit App UI
# --------------------

def sidebar_controls(n_racks: int) -> Dict:
    """
    Sidebar controls for global config: number of racks, transfer fraction, thresholds, random seed
    """
    st.sidebar.header("Simulation settings")
    n = st.sidebar.slider("Number of racks", min_value=20, max_value=30, value=n_racks)
    seed = st.sidebar.number_input("Random seed", value=RANDOM_SEED, step=1)
    transfer_pct = st.sidebar.slider("Transfer fraction (per hotspot)", min_value=0.05, max_value=0.5, value=TRANSFE R_PERCENT if 'TRANSFE R_PERCENT' not in globals() else TRANSFER_PERCENT, step=0.05)  # fallback
    # Note: Above line tries to handle variable naming; simpler approach below if glitch
    if 'TRANSFER_PERCENT' in globals():
        transfer_pct = st.sidebar.slider("Transfer fraction (per hotspot)", min_value=0.05, max_value=0.5, value=TRANSFER_PERCENT, step=0.05)
    temp_thresh = st.sidebar.slider("Temperature threshold for candidate receivers (°C)", min_value=40, max_value=75, value=60)
    cpu_thresh = st.sidebar.slider("CPU threshold for candidate receivers (%)", min_value=30, max_value=80, value=60)
    return {"n": n, "seed": int(seed), "transfer_pct": float(transfer_pct), "temp_thresh": float(temp_thresh), "cpu_thresh": float(cpu_thresh)}

def main():
    st.set_page_config(layout="wide", page_title="Data Center Hotspot Predictor & Optimizer")
    st.title("🔧 Data Center Hotspot Predictor & Workload Optimizer (Prototype)")
    st.markdown(
        """
        **What this app does**
        - Simulates a small data center (20–30 racks) with correlated CPU, temperature, network, and power.
        - Trains a Random Forest model to predict hotspots (CPU > 80% and Temp > 70°C).
        - Suggests simple workload redistribution actions to reduce hotspots and projected max temperature/power.
        - Interactive Streamlit dashboard with heatmap, sliders per rack, dynamic predictions, and an 'Apply AI Recommendation' action.
        """
    )

    # Sidebar parameters
    st.sidebar.title("Controls")
    n_racks = st.sidebar.slider("Initial number of racks", 20, 30, DEFAULT_RACK_COUNT)
    seed = st.sidebar.number_input("Random seed", value=RANDOM_SEED, step=1)
    transfer_fraction = st.sidebar.slider("Transfer fraction per hotspot", 0.05, 0.5, TRANSFER_PERCENT, step=0.05)
    cpu_receiver_thresh = st.sidebar.slider("Receiver CPU < (target %) ", 30, 80, 60)
    temp_receiver_thresh = st.sidebar.slider("Receiver Temp < (°C) ", 40, 75, 60)

    # Generate baseline dataset
    df = generate_racks(n_racks, seed)

    # Train model (cached)
    with st.spinner("Training model..."):
        model, scaler, eval_df = train_model(df, random_seed=seed)

    # Initial predictions for the current (simulated) state
    df_pred = predict_all(df, model, scaler)

    # Show model evaluation metrics
    st.subheader("Model Evaluation (test split)")
    # Compute metrics from eval_df returned by train_model
    try:
        y_true = eval_df["Actual"]
        y_pred = eval_df["Predicted"]
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        st.write(f"**Accuracy (on test set):** {acc:.3f}")
        st.write("**Confusion Matrix (test set)**")
        st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    except Exception as e:
        st.write("Could not compute evaluation metrics:", e)

    # Layout: left column controls + rack sliders, right column heatmap & metrics
    left_col, right_col = st.columns([1, 2])

    # LEFT: Sliders per rack (dynamically created). We allow per-rack CPU slider updates.
    with left_col:
        st.header("Rack controls")
        st.write("Use the sliders to tweak CPU load per rack and see AI predictions update in real time.")
        st.markdown("**Tip:** You can change multiple sliders before hitting the blue button to re-run predictions.")

        # We'll present sliders in groups/columns to avoid a very long vertical list.
        # Create a container to hold slider states.
        slider_container = st.container()
        # Keep track of sliders values in a dict
        cpu_updates = {}
        # For many racks, display in multiple columns
        slider_cols = st.columns(3)
        for idx in range(len(df_pred)):
            col_idx = idx % 3
            with slider_cols[col_idx]:
                rid = df_pred.iloc[idx]["Rack_ID"]
                default_cpu = float(df_pred.iloc[idx]["CPU_load"])
                # Create a unique key for each slider
                key = f"cpu_slider_{rid}"
                cpu_val = st.slider(f"{rid} CPU (%)", 0.0, 100.0, default_cpu, key=key)
                cpu_updates[rid] = cpu_val

        # Action: Apply CPU updates to simulated DataFrame and re-run model predictions
        if st.button("Update CPU loads and re-run AI predictions"):
            # Apply cpu_updates to df_pred
            for i in range(len(df_pred)):
                rid = df_pred.iloc[i]["Rack_ID"]
                df_pred.at[i, "CPU_load"] = float(cpu_updates[rid])
            # Recompute Temperature and Power using synthetic model (so UI remains consistent)
            df_pred["Temperature"] = np.clip(
                TEMP_BASE + TEMP_CPU_COEFF * df_pred["CPU_load"] + TEMP_NET_COEFF * df_pred["Network_usage"]
                + np.random.normal(0, TEMP_NOISE_STD, size=len(df_pred)), 15.0, 95.0)
            df_pred["Power_consumption"] = np.clip(
                POWER_BASE + POWER_CPU_COEFF * df_pred["CPU_load"] + POWER_NET_COEFF * df_pred["Network_usage"]
                + np.random.normal(0, POWER_NOISE_STD, size=len(df_pred)), 0.05, 15.0)

            # Re-run predictions using model
            df_pred = predict_all(df_pred, model, scaler)
            st.success("Predictions updated.")

    # RIGHT: Heatmap and summary stats
    with right_col:
        st.header("Data Center Heatmap & AI Predictions")
        st.markdown("Heatmap shows **Temperature** per rack. Each cell annotation shows Rack_ID, temperature, and AI risk (probability).")

        fig = plot_temperature_heatmap(df_pred, value_col="Temperature", risk_col="Pred_Proba", cols=GRID_COLS,
                                       title="Current Rack Temperatures & Risk")
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics before recommendation
        st.subheader("Current summary")
        num_hotspots = int(df_pred["Predicted"].sum())
        total_power = float(df_pred["Power_consumption"].sum())
        max_temp = float(df_pred["Temperature"].max())
        st.metric("Predicted hotspots", value=f"{num_hotspots}", delta=None)
        st.metric("Total power consumption (kW)", value=f"{total_power:.2f}")
        st.metric("Max rack temperature (°C)", value=f"{max_temp:.2f}")

        # Show top risky racks table
        st.write("Top risky racks (by predicted hotspot probability):")
        top_risky = df_pred.sort_values("Pred_Proba", ascending=False).head(6)[
            ["Rack_ID", "CPU_load", "Temperature", "Pred_Proba", "Predicted"]]
        st.dataframe(top_risky.reset_index(drop=True))

    # Recommendation Action (global, centered)
    st.markdown("---")
    st.header("AI Recommendation Engine")
    st.write(
        f"This recommendation will attempt to move **{transfer_fraction*100:.0f}%** of each hotspot's CPU load "
        "to under-utilized racks (prioritize racks with CPU and Temp below thresholds)."
    )

    # Display current candidate counts
    candidate_count = len(df_pred[(df_pred["CPU_load"] < cpu_receiver_thresh) & (df_pred["Temperature"] < temp_receiver_thresh)])
    st.write(f"Candidate receiver racks (CPU < {cpu_receiver_thresh}% and Temp < {temp_receiver_thresh}°C): **{candidate_count}**")

    # Button to apply AI recommendation
    if st.button("Apply AI Recommendation"):
        # Run recommendation engine on df_pred (which currently reflects any slider changes)
        df_projected, plan = recommend_redistribution(df_pred, transfer_fraction, cpu_receiver_thresh, temp_receiver_thresh)

        # Projected summary
        projected_hotspots = int(df_projected["Projected_Hotspot"].sum())
        projected_total_power = float(df_projected["Projected_Power"].sum())
        projected_max_temp = float(df_projected["Projected_Temperature"].max())

        # Show plan (first 10 actions)
        st.subheader("Redistribution plan (sample actions)")
        if len(plan) == 0:
            st.info("No feasible transfers found (no suitable receivers or transfers too small). Consider lowering thresholds or increasing transfer fraction.")
        else:
            plan_df = pd.DataFrame(plan)
            st.dataframe(plan_df.head(12))

        # Display projected improvements
        st.subheader("Projected improvements after applying recommendation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Hotspots (before → after)", f"{int(df_pred['Predicted'].sum())} → {projected_hotspots}")
        col2.metric("Total power (kW) (before → after)", f"{df_pred['Power_consumption'].sum():.2f} → {projected_total_power:.2f}",
                    delta=f"{projected_total_power - df_pred['Power_consumption'].sum():+.2f}")
        col3.metric("Max temp (°C) (before → after)", f"{df_pred['Temperature'].max():.2f} → {projected_max_temp:.2f}",
                    delta=f"{projected_max_temp - df_pred['Temperature'].max():+.2f}")

        # Show projected heatmap (right column)
        st.subheader("Projected rack temperatures (after redistribution)")
        fig2 = plot_temperature_heatmap(df_projected, value_col="Projected_Temperature", risk_col="Pred_Proba", cols=GRID_COLS,
                                        title="Projected Rack Temperatures & Risk (after redistribution)")
        st.plotly_chart(fig2, use_container_width=True)

        # Optional: Recompute AI predictions on the projected state to show residual risk
        # We'll modify the DataFrame to include projected CPU/temp/power in features and re-evaluate model predictions
        df_for_model = df_projected.copy()
        # Overwrite the primary features used by the model with projected ones
        df_for_model["CPU_load"] = df_for_model["Projected_CPU"]
        df_for_model["Temperature"] = df_for_model["Projected_Temperature"]
        df_for_model["Power_consumption"] = df_for_model["Projected_Power"]
        df_after_pred = predict_all(df_for_model, model, scaler)

        st.subheader("Residual AI predictions (after redistribution)")
        after_top_risky = df_after_pred.sort_values("Pred_Proba", ascending=False).head(6)[
            ["Rack_ID", "Projected_CPU", "Projected_Temperature", "Pred_Proba", "Predicted"]]
        st.dataframe(after_top_risky.reset_index(drop=True))

        # Optionally allow the user to accept the projected state as the new baseline
        if st.button("Accept projected state as new baseline"):
            # Commit changes to df_pred so the UI shows updated CPU/temp/power
            # We will map back the projected values into df_pred (match by Rack_ID)
            id_to_idx = {rid: i for i, rid in enumerate(df_pred["Rack_ID"].tolist())}
            for i, row in df_projected.iterrows():
                rid = row["Rack_ID"]
                if rid in id_to_idx:
                    idx = id_to_idx[rid]
                    df_pred.at[idx, "CPU_load"] = float(row["Projected_CPU"])
                    df_pred.at[idx, "Temperature"] = float(row["Projected_Temperature"])
                    df_pred.at[idx, "Power_consumption"] = float(row["Projected_Power"])
            # Re-run predictions to update predicted/hotspot columns
            df_pred = predict_all(df_pred, model, scaler)
            st.success("Projected state accepted. Baseline updated.")

    # Footer: show raw simulated dataset option
    with st.expander("Show full simulated dataset"):
        st.dataframe(df_pred.reset_index(drop=True))

    st.markdown("""
    ---
    **Notes & Implementation Details:**
    - The "physics" linking CPU -> Temperature/Power is a linear, simplified model for prototype purposes.
    - RandomForest is used because it trains quickly and provides robust probabilities.
    - Recommendation engine uses a conservative, explainable heuristic (transfer a fraction of CPU from hotspots to low-utilization racks).
    - This prototype is intentionally simple but extensible: you can replace the models, add constraints (rack affinity, cable/power limits), or use optimization solvers (e.g., linear/integer programming) for more realistic redistribution.
    """)

if __name__ == "__main__":
    main()
