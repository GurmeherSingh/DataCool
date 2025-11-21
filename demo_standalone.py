"""
Standalone Demo Script (No Streamlit Required)

Quick test of the enhanced data center simulation.
Runs from command line and prints results.

Usage: python demo_standalone.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import math

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_RACKS = 24
GRID_COLS = 6

# Physics constants
TEMP_AMBIENT = 20.0
TEMP_CPU_COEFF = 0.45
TEMP_NET_COEFF = 0.01
TEMP_ADJACENCY_COEFF = 0.08
POWER_BASE = 0.5
POWER_CPU_COEFF = 0.06
POWER_NET_COEFF = 0.001

HOT_CPU_THRESHOLD = 80.0
HOT_TEMP_THRESHOLD = 70.0
MIGRATION_COST_PER_PCT = 0.5
MAX_MIGRATION_COST = 100.0

def get_rack_position(rack_idx, cols=GRID_COLS):
    return (rack_idx // cols, rack_idx % cols)

def get_adjacent_racks(rack_idx, n_racks, cols=GRID_COLS):
    row, col = get_rack_position(rack_idx, cols)
    rows = math.ceil(n_racks / cols)
    adjacent = []
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            adj_idx = new_row * cols + new_col
            if adj_idx < n_racks:
                adjacent.append(adj_idx)
    return adjacent

def assign_thermal_zones(n_racks):
    zones = np.zeros(n_racks, dtype=int)
    for i in range(n_racks):
        row, col = get_rack_position(i)
        zones[i] = 0 if row < 1 else (1 if row < 3 else 2)
    return zones

def compute_temperature(cpu_loads, network_usage, n_racks, zones, base_offsets):
    temp = (TEMP_AMBIENT + TEMP_CPU_COEFF * cpu_loads + 
            TEMP_NET_COEFF * network_usage + base_offsets)
    
    zone_penalties = np.array([0.0, 2.5, 5.0])
    temp += zone_penalties[zones]
    
    for iteration in range(3):
        temp_influence = np.zeros(n_racks)
        for i in range(n_racks):
            adjacent = get_adjacent_racks(i, n_racks)
            if adjacent:
                avg_neighbor_temp = np.mean(temp[adjacent])
                temp_influence[i] = TEMP_ADJACENCY_COEFF * (avg_neighbor_temp - temp[i])
        temp += temp_influence * 0.5
    
    temp += np.random.normal(0, 1.0, size=n_racks)
    return np.clip(temp, 15.0, 95.0)

def generate_racks(n_racks, seed=RANDOM_SEED):
    np.random.seed(seed)
    
    cpu_low = np.random.beta(2, 5, int(n_racks * 0.5)) * 50
    cpu_medium = np.random.beta(3, 3, int(n_racks * 0.3)) * 60 + 30
    cpu_high = np.random.beta(5, 2, n_racks - int(n_racks * 0.8)) * 50 + 50
    cpu_load = np.concatenate([cpu_low, cpu_medium, cpu_high])
    np.random.shuffle(cpu_load)
    cpu_load = np.clip(cpu_load + np.random.normal(0, 3, n_racks), 0, 100)
    
    network_base = np.random.exponential(120, n_racks)
    network_cpu_bonus = cpu_load * 2.5
    network_usage = np.clip(network_base + network_cpu_bonus + 
                           np.random.normal(0, 50, n_racks), 0, 1000)
    
    zones = assign_thermal_zones(n_racks)
    base_temp_offsets = np.random.normal(0, 1.5, n_racks)
    temperature = compute_temperature(cpu_load, network_usage, n_racks, 
                                     zones, base_temp_offsets)
    
    power = (POWER_BASE + POWER_CPU_COEFF * cpu_load + 
             POWER_NET_COEFF * network_usage)
    power = np.clip(power, 0.1, 15.0)
    
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
    
    df["Hotspot"] = ((df["CPU_load"] > HOT_CPU_THRESHOLD) & 
                     (df["Temperature"] > HOT_TEMP_THRESHOLD)).astype(int)
    
    return df

def train_and_evaluate(df):
    FEATURES = ["CPU_load", "Temperature", "Network_usage", 
                "Power_consumption", "Thermal_Zone"]
    X = df[FEATURES].values
    y = df["Hotspot"].values
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED
        )
    
    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc, FEATURES

def predict_all(df, model, features):
    X = df[features].values
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    df_copy = df.copy()
    df_copy["Pred_Proba"] = proba
    df_copy["Predicted"] = pred
    return df_copy

def optimize_workload(df):
    n_racks = len(df)
    hotspots = df[df["Predicted"] == 1].index.tolist()
    candidates = df[df["Predicted"] == 0].index.tolist()
    
    if not hotspots or not candidates:
        return df.copy(), [], {}
    
    df_opt = df.copy()
    df_opt["Optimized_CPU"] = df_opt["CPU_load"].copy()
    
    transfers = []
    total_cost = 0.0
    
    hotspots_sorted = df.loc[hotspots].sort_values("Temperature", ascending=False).index.tolist()
    candidates_sorted = df.loc[candidates].sort_values("CPU_load").index.tolist()
    
    for hot_idx in hotspots_sorted:
        if total_cost >= MAX_MIGRATION_COST:
            break
        
        hot_cpu = df_opt.at[hot_idx, "Optimized_CPU"]
        target_reduction = min(30.0, hot_cpu - 70.0)
        
        if target_reduction <= 0:
            continue
        
        moved = 0.0
        for cand_idx in candidates_sorted:
            if moved >= target_reduction or total_cost >= MAX_MIGRATION_COST:
                break
            
            cand_cpu = df_opt.at[cand_idx, "Optimized_CPU"]
            available_capacity = min(85.0 - cand_cpu, 95.0 - cand_cpu)
            
            if available_capacity <= 1.0:
                continue
            
            hot_pos = (df.at[hot_idx, "Row"], df.at[hot_idx, "Col"])
            cand_pos = (df.at[cand_idx, "Row"], df.at[cand_idx, "Col"])
            distance = abs(hot_pos[0] - cand_pos[0]) + abs(hot_pos[1] - cand_pos[1])
            distance_penalty = 1.0 + 0.2 * distance
            
            chunk = min(target_reduction - moved, available_capacity, 20.0)
            chunk_cost = chunk * MIGRATION_COST_PER_PCT * distance_penalty
            
            if total_cost + chunk_cost > MAX_MIGRATION_COST:
                chunk = (MAX_MIGRATION_COST - total_cost) / (MIGRATION_COST_PER_PCT * distance_penalty)
                chunk = max(0.0, min(chunk, available_capacity))
            
            if chunk < 1.0:
                continue
            
            df_opt.at[hot_idx, "Optimized_CPU"] -= chunk
            df_opt.at[cand_idx, "Optimized_CPU"] += chunk
            moved += chunk
            total_cost += chunk_cost
            
            transfers.append({
                "From": df.at[hot_idx, "Rack_ID"],
                "To": df.at[cand_idx, "Rack_ID"],
                "CPU_Moved": round(chunk, 2),
                "Cost": round(chunk_cost, 2)
            })
    
    # Recompute temps
    zones = df_opt["Thermal_Zone"].values
    base_offsets = np.zeros(n_racks)
    df_opt["Optimized_Temp"] = compute_temperature(
        df_opt["Optimized_CPU"].values,
        df_opt["Network_usage"].values,
        n_racks, zones, base_offsets
    )
    
    metrics = {
        "total_cost": round(total_cost, 2),
        "transfers": len(transfers),
        "hotspots_before": int(df["Predicted"].sum()),
        "hotspots_after": int(((df_opt["Optimized_CPU"] > HOT_CPU_THRESHOLD) & 
                               (df_opt["Optimized_Temp"] > HOT_TEMP_THRESHOLD)).sum()),
        "max_temp_before": float(df["Temperature"].max()),
        "max_temp_after": float(df_opt["Optimized_Temp"].max())
    }
    
    return df_opt, transfers, metrics

def main():
    print("=" * 70)
    print(" 🚀 Data Center AI Optimizer - Standalone Demo")
    print("=" * 70)
    print()
    
    # Generate data
    print(f"📊 Generating {N_RACKS} racks...")
    df = generate_racks(N_RACKS)
    print(f"   ✓ Generated with {df['Hotspot'].sum()} actual hotspots")
    print()
    
    # Train model
    print("🧠 Training AI model (HistGradientBoosting)...")
    model, acc, features = train_and_evaluate(df)
    print(f"   ✓ Model trained with {acc:.1%} accuracy")
    print()
    
    # Predict
    print("🔮 Making predictions...")
    df_pred = predict_all(df, model, features)
    print(f"   ✓ Predicted {df_pred['Predicted'].sum()} hotspots")
    print()
    
    # Show current state
    print("📈 CURRENT STATE:")
    print("-" * 70)
    print(f"   Hotspots detected:     {int(df_pred['Predicted'].sum())}")
    print(f"   Max temperature:       {df_pred['Temperature'].max():.1f}°C")
    print(f"   Avg CPU load:          {df_pred['CPU_load'].mean():.1f}%")
    print(f"   Total power:           {df_pred['Power_consumption'].sum():.2f} kW")
    print()
    
    # Show top risky racks
    print("⚠️  TOP 5 RISKY RACKS:")
    print("-" * 70)
    top_risky = df_pred.sort_values("Pred_Proba", ascending=False).head(5)
    for _, row in top_risky.iterrows():
        print(f"   {row['Rack_ID']}: CPU={row['CPU_load']:.0f}%, "
              f"Temp={row['Temperature']:.1f}°C, Risk={row['Pred_Proba']:.2f}, "
              f"Zone={row['Thermal_Zone']}")
    print()
    
    # Optimize
    print("⚡ Running AI-powered optimization...")
    df_opt, transfers, metrics = optimize_workload(df_pred)
    print(f"   ✓ Completed {metrics['transfers']} transfers")
    print(f"   ✓ Cost: {metrics['total_cost']:.1f} / {MAX_MIGRATION_COST} units")
    print()
    
    # Show results
    print("📊 OPTIMIZATION RESULTS:")
    print("=" * 70)
    print(f"   Hotspots:  {metrics['hotspots_before']} → {metrics['hotspots_after']} "
          f"({metrics['hotspots_before'] - metrics['hotspots_after']} eliminated)")
    print(f"   Max temp:  {metrics['max_temp_before']:.1f}°C → {metrics['max_temp_after']:.1f}°C "
          f"({metrics['max_temp_after'] - metrics['max_temp_before']:.1f}°C)")
    print()
    
    # Show migration plan
    if transfers:
        print("🔄 MIGRATION PLAN:")
        print("-" * 70)
        for i, t in enumerate(transfers[:10], 1):
            print(f"   {i}. {t['From']} → {t['To']}: {t['CPU_Moved']:.1f}% CPU "
                  f"(cost: {t['Cost']:.1f})")
        if len(transfers) > 10:
            print(f"   ... and {len(transfers) - 10} more transfers")
    print()
    
    print("=" * 70)
    print("✅ Demo complete! Run 'streamlit run datacenter_ai_enhanced.py' for full UI")
    print("=" * 70)

if __name__ == "__main__":
    main()

