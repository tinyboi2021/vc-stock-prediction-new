"""
model_diagnostics.py
====================
Diagnose feature dimension mismatches between saved models and inference data.

This script:
1. Loads all saved model checkpoints
2. Extracts their input_dim from the config
3. Checks scaler dimensions
4. Provides guidance on fixing dimension mismatches

Usage:
    python model_diagnostics.py
"""

import torch
import joblib
import os
import pandas as pd
from pathlib import Path

MODEL_DIR = "./Saved_Models"
SCALER_DIR = "./Saved_Scalers"
RESULTS_CSV = "./Paper_Replication_Results.csv"

print("="*80)
print("MODEL DIMENSION DIAGNOSTICS")
print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check all model checkpoints
# ─────────────────────────────────────────────────────────────────────────────

print("\n1. CHECKING MODEL CHECKPOINTS\n")
print(f"{'Model File':<50} {'Input Dim':<10} {'Scenario':<20}")
print("-"*80)

model_files = sorted(Path(MODEL_DIR).glob("*.pth"))

model_dims = {}
for model_path in model_files:
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'config' in checkpoint:
            input_dim = checkpoint['config'].get('input_dim', 'N/A')
            scenario = checkpoint['config'].get('scenario', 'Unknown')
            
            model_dims[model_path.name] = {
                'input_dim': input_dim,
                'scenario': scenario,
                'config': checkpoint['config']
            }
            
            print(f"{model_path.name:<50} {input_dim:<10} {scenario:<20}")
        else:
            print(f"{model_path.name:<50} {'NO CONFIG':<10} {'Unknown':<20}")
            
    except Exception as e:
        print(f"{model_path.name:<50} ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Check scaler dimensions
# ─────────────────────────────────────────────────────────────────────────────

print("\n2. CHECKING SCALER DIMENSIONS\n")
print(f"{'Scaler File':<30} {'Target Dim':<12} {'Covariate Dim':<15} {'Total Features':<15}")
print("-"*80)

scaler_files = sorted(Path(SCALER_DIR).glob("*.pkl"))

scaler_dims = {}
for scaler_path in scaler_files:
    try:
        scalers = joblib.load(scaler_path)
        
        target_scaler = scalers.get('target')
        cov_scaler = scalers.get('cov')
        
        target_dim = target_scaler.n_features_in_ if target_scaler else 0
        cov_dim = cov_scaler.n_features_in_ if cov_scaler else 0
        total = target_dim + cov_dim
        
        scaler_dims[scaler_path.name] = {
            'target': target_dim,
            'cov': cov_dim,
            'total': total
        }
        
        print(f"{scaler_path.name:<30} {target_dim:<12} {cov_dim:<15} {total:<15}")
        
    except Exception as e:
        print(f"{scaler_path.name:<30} ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Check results CSV for expected dimensions
# ─────────────────────────────────────────────────────────────────────────────

print("\n3. CHECKING RESULTS CSV\n")

if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    
    print(f"Total rows: {len(df)}")
    print(f"Scenarios: {df['Scenario'].unique().tolist()}")
    print(f"Horizons: {sorted(df['Horizon'].unique().tolist())}")
    print(f"Models: {df['Model'].unique().tolist()}")
    
    # Group by scenario and show counts
    print("\nBreakdown by scenario:")
    for scenario in df['Scenario'].unique():
        count = len(df[df['Scenario'] == scenario])
        print(f"  {scenario:<25} : {count} configurations")
else:
    print(f"Results CSV not found at {RESULTS_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Analyze dimension mismatches
# ─────────────────────────────────────────────────────────────────────────────

print("\n4. DIMENSION ANALYSIS\n")
print("="*80)

# Group models by input_dim
dim_groups = {}
for model_name, info in model_dims.items():
    dim = info['input_dim']
    if dim not in dim_groups:
        dim_groups[dim] = []
    dim_groups[dim].append((model_name, info['scenario']))

print("Models grouped by input dimension:")
for dim in sorted(dim_groups.keys()):
    models = dim_groups[dim]
    print(f"\n  Input Dim = {dim}  ({len(models)} models)")
    for model_name, scenario in models[:5]:  # Show first 5
        print(f"    - {model_name:<45} ({scenario})")
    if len(models) > 5:
        print(f"    ... and {len(models)-5} more")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Provide recommendations
# ─────────────────────────────────────────────────────────────────────────────

print("\n5. RECOMMENDATIONS\n")
print("="*80)

# Determine expected dimensions based on your dataset
print("\nExpected feature dimensions:")
print("  33 raw features (without sentiment columns)")
print("  + 4 technical indicators (SMA, RSI, OBV, ADX)")
print("  ─────────────────────────────────────────────")
print("  = 37 total for 'Without Sentiment' scenario")
print()
print("  37 raw features (with sentiment columns)")
print("  + 4 technical indicators")
print("  ─────────────────────────────────────────────")
print("  = 41 total for 'With Sentiment' scenario")

# Check actual dimensions
if model_dims:
    actual_dims = set(info['input_dim'] for info in model_dims.values() if info['input_dim'] != 'N/A')
    print(f"\nActual model dimensions found: {sorted(actual_dims)}")
    
    if 33 in actual_dims or 37 in actual_dims:
        print("\n⚠️  MISMATCH DETECTED!")
        print("Your models appear to have been trained WITHOUT technical indicators.")
        print("But your inference code is ADDING them.")
        print("\nSOLUTION OPTIONS:")
        print("  A) Retrain models with technical indicators (recommended)")
        print("  B) Remove technical indicator generation from inference code")
        print("  C) Use feature count that matches your training")

if scaler_dims:
    scaler_totals = set(info['total'] for info in scaler_dims.values())
    print(f"\nScaler dimensions found: {sorted(scaler_totals)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Generate feature count mapping
# ─────────────────────────────────────────────────────────────────────────────

print("\n6. FEATURE COUNT MAPPING FOR YOUR DATASET\n")
print("="*80)

print("""
Based on your training code, here's what each scenario should have:

Scenario: "With Sentiment"
  Raw features before technical indicators:
    - Close, High, Low, Volume                    (4 OHLCV)
    - Gold_Close, CPI, Interest Rate, etc.        (29 economic/market/tech)
    - sentiment_score, sentiment_std, etc.        (4 sentiment)
    ───────────────────────────────────────────── 
    Total raw: 37 features
  
  After add_technical_indicators():
    + SMA, RSI, OBV, ADX                          (4 technical)
    ─────────────────────────────────────────────
    Total: 41 features

Scenario: "Without Sentiment"  
  Raw features before technical indicators:
    - Close, High, Low, Volume                    (4 OHLCV)
    - Gold_Close, CPI, Interest Rate, etc.        (29 economic/market/tech)
    ─────────────────────────────────────────────
    Total raw: 33 features
  
  After add_technical_indicators():
    + SMA, RSI, OBV, ADX                          (4 technical)
    ─────────────────────────────────────────────
    Total: 37 features

YOUR ISSUE: Models have input_dim=33, suggesting they were trained
WITHOUT technical indicators being added, OR your training was done
with the old 7-feature mock data.
""")

print("\n7. IMMEDIATE FIX\n")
print("="*80)
print("""
To fix the dimension mismatch, modify your generate_mock_live_data() function:

Option A: Match what models were trained with (NO technical indicators)
─────────────────────────────────────────────────────────────────────────
# Remove this line from generate_mock_live_data():
# df = add_technical_indicators(df, ...)

# Manually add these columns with dummy values instead:
df['SMA'] = df['Close']  # Dummy SMA
df['RSI'] = 50.0         # Dummy RSI  
df['OBV'] = 0.0          # Dummy OBV
df['ADX'] = 0.25         # Dummy ADX

Option B: Retrain with correct feature pipeline (RECOMMENDED)
─────────────────────────────────────────────────────────────
# Ensure your training code:
1. Loads raw 33/37 features
2. Calls add_technical_indicators() AFTER train/test split
3. Results in input_dim = 37 (without sentiment) or 41 (with sentiment)
4. Re-run full training pipeline

Then your inference will work as-is.
""")

print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)