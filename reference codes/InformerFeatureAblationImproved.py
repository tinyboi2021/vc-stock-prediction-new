"""
Feature Ablation Study using Informer Model
===========================================
Model: Informer_FullAttn_NoDistil
Horizon: 16 days
Dataset: aapl_sentiment_fixed_merged_dataset.csv

This script tests the contribution of different feature groups using the best-performing
model configuration found in previous experiments.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
import warnings
import optuna
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import the specific Informer model
from informerMLopsUpdated import Informer

# ==============================
# CONFIGURATION
# ==============================

class Config:
    """Centralized configuration"""
    
    # ============ PROFILE SETTINGS ============
    RUN_PROFILE = "SLOW"  # Options: "FAST" (testing), "SLOW" (production)
    
    # ============ PATHS ============
    DATA_PATH = "./aapl_sentiment_fixed_merged_dataset.csv"
    MODEL_SAVE_DIR = "./Saved_Models_Ablation"
    RESULTS_CSV = "./feature_ablation_results_H16.csv"
    
    # ============ DEVICE ============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============ SEED ============
    RANDOM_SEED = 42
    
    # ============ EXPERIMENT FLAGS ============
    PREDICT_RETURNS = True              
    USE_DIRECTIONAL_LOSS = True         
    USE_ENSEMBLE = True                 
    
    # ============ MODEL SPECIFIC (Informer_FullAttn_NoDistil) ============
    DISTIL = False
    PROB_SPARSE = False
    DEC_ONLY = False
    GEN_DEC = True
    
    # ============ PROFILE-SPECIFIC SETTINGS ============
    if RUN_PROFILE.upper() == "FAST":
        EPOCHS = 50
        PATIENCE = 10
        N_TRIALS = 5  # Reduced trials for ablation
        LOOKBACK_WINDOWS = [96] # Fixed optimal lookback
        PREDICTION_HORIZONS = [16] # Fixed Horizon
        D_MODEL = 64
        N_HEADS = 8
        BATCH_SIZE = 32
        DROPOUT = 0.1
        N_ENSEMBLE_MODELS = 2
        
    elif RUN_PROFILE.upper() == "SLOW":
        EPOCHS = 200
        PATIENCE = 20
        N_TRIALS = 10 # Trials per scenario
        LOOKBACK_WINDOWS = [96] # Best found lookback
        PREDICTION_HORIZONS = [16] # Target Horizon
        D_MODEL = 64 # From best config if available, or standard
        N_HEADS = 8
        BATCH_SIZE = 32
        DROPOUT = 0.1
        N_ENSEMBLE_MODELS = 3
    
    # ============ DATA SPLIT ============
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.85
    
    # ============ TRAINING SETTINGS ============
    GRADIENT_CLIP = 1.0
    WEIGHT_DECAY = 1e-4
    DIRECTION_LOSS_WEIGHT = 1.5
    MSE_LOSS_WEIGHT = 1.0

# Initialize
config = Config()
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

print(f"{'='*70}")
print(f"FEATURE ABLATION STUDY - INFORMER FULL ATTN NO DISTIL")
print(f"{'='*70}")
print(f"Horizon: {config.PREDICTION_HORIZONS[0]} days")
print(f"Model: Informer (Distil={config.DISTIL}, ProbSparse={config.PROB_SPARSE})")
print(f"{'='*70}\\n")

# ==============================
# FEATURE GROUPS
# ==============================

FEATURE_GROUPS = {
    'baseline': ['Close', 'High', 'Low', 'Volume', 'SMA', 'RSI', 'OBV', 'ADX'],
    'macro_economic': [
        'CPI (Inflation)', 
        'Interest Rate / Fed Funds Rate', 
        'Bond Yields (10Y US Treasury)', 
        'Money Supply (M2)', 
        'Unemployment Rate'
    ],
    'market_risk': [
        'VIX Index', 
        'USD Index (DXY)', 
        'USD/CNY', 
        'USD/EUR', 
        'Gold_Close'
    ],
    'ai_hardware': [
        'GPU price index', 
        'Semiconductor index', 
        'Memory price (DRAM)', 
        'HBM price trend'
    ],
    'energy_infra': [
        'Electricity price (industrial)', 
        'Natural gas price', 
        'Carbon price / energy regulation', 
        'US Power Grid Load Index'
    ],
    'ai_investment': [
        'Global AI investment index', 
        'Tech CAPEX index', 
        'AI regulation events (binary)', 
        'Government AI funding'
    ],
    'news_sentiment': [
        'sentiment_score', 
        'sentiment_std', 
        'article_volume', 
        'high_confidence_ratio'
    ],
    'staleness': [
        'CPI (Inflation)_days_stale',
        'Money Supply (M2)_days_stale',
        'Unemployment Rate_days_stale'
    ]
}

ABLATION_SCENARIOS = {
    '1_Baseline_Only': ['baseline'],
    '2_Baseline+Macro': ['baseline', 'macro_economic'],
    '3_Baseline+Macro+Risk': ['baseline', 'macro_economic', 'market_risk'],
    '4_Baseline+Macro+Risk+AIHardware': ['baseline', 'macro_economic', 'market_risk', 'ai_hardware'],
    '5_Baseline+Macro+Risk+AIHardware+Energy': ['baseline', 'macro_economic', 'market_risk', 'ai_hardware', 'energy_infra'],
    '6_All_NoNews': ['baseline', 'macro_economic', 'market_risk', 'ai_hardware', 'energy_infra', 'ai_investment', 'staleness'],
    '7_All_Features': ['baseline', 'macro_economic', 'market_risk', 'ai_hardware', 
                      'energy_infra', 'ai_investment', 'news_sentiment', 'staleness'],
}

def get_feature_columns(scenario_name: str) -> List[str]:
    """Get feature columns for a given scenario"""
    groups = ABLATION_SCENARIOS[scenario_name]
    cols = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    # Remove duplicates while preserving order
    seen = set()
    cols = [x for x in cols if not (x in seen or seen.add(x))]
    return cols

# ==============================
# UTILITY FUNCTIONS
# ==============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(config.RANDOM_SEED)

def append_result_to_csv(result_dict: Dict):
    df_new = pd.DataFrame([result_dict])
    header = not os.path.exists(config.RESULTS_CSV)
    df_new.to_csv(config.RESULTS_CSV, mode="a", header=header, index=False)

def time_features(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.vstack([
        dates.month, 
        dates.day, 
        dates.dayofweek,
        dates.hour if hasattr(dates, 'hour') else np.zeros(len(dates))
    ]).transpose()

# ==============================
# TECHNICAL INDICATORS
# ==============================

def calculate_technical_indicators_post_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'High' not in df.columns or 'Low' not in df.columns or 'Volume' not in df.columns:
        return df
    
    # SMA
    df['SMA'] = df['Close'].rolling(window=20, min_periods=1).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # ADX
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=1).mean() / (atr + 1e-10))
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14, min_periods=1).mean() / (atr + 1e-10)))
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-10)) * 100
    df['ADX'] = dx.ewm(alpha=1/14, min_periods=1).mean()
    
    return df

# ==============================
# DATA LOADING
# ==============================

def load_and_split_data(target_col: str = 'Close', 
                        feature_columns: Optional[List[str]] = None) -> Tuple:
    print(f"\\n  Loading data from: {config.DATA_PATH}")
    
    try:
        df_raw = pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        print(f"  ❌ File not found: {config.DATA_PATH}")
        raise
    
    if "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    else:
        df_raw["Date"] = pd.to_datetime(df_raw.index)
    
    df_raw.set_index('Date', inplace=True)
    df_raw = df_raw.sort_index()

    if feature_columns is not None:
        feature_cols = [c for c in feature_columns if c in df_raw.columns]
    else:
        feature_cols = [c for c in df_raw.columns]
    
    if target_col not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset")
    
    # Returns Prediction
    if config.PREDICT_RETURNS:
        df_raw['Target'] = df_raw[target_col].pct_change()
        target_col_name = 'Target'
    else:
        target_col_name = target_col
    
    if target_col_name not in feature_cols:
        feature_cols = [target_col_name] + feature_cols
    else:
        feature_cols.remove(target_col_name)
        feature_cols = [target_col_name] + feature_cols
    
    df_full = df_raw[feature_cols].copy()
    
    # Forward fill only
    df_full = df_full.ffill().fillna(0)
    df_full = df_full.dropna()
    
    date_index = df_full.index
    n = len(df_full)
    val_start = int(n * config.TRAIN_SPLIT)
    test_start = int(n * config.VAL_SPLIT)
    
    train_df = df_full.iloc[:val_start].copy()
    val_df = df_full.iloc[val_start:test_start].copy()
    test_df = df_full.iloc[test_start:].copy()
    
    train_df = calculate_technical_indicators_post_split(train_df)
    val_df = calculate_technical_indicators_post_split(val_df)
    test_df = calculate_technical_indicators_post_split(test_df)
    
    train_df = train_df.ffill().fillna(0)
    val_df = val_df.ffill().fillna(0)
    test_df = test_df.ffill().fillna(0)
    
    train_vals = train_df.values
    val_vals = val_df.values
    test_vals = test_df.values
    
    train_marks = time_features(train_df.index)
    val_marks = time_features(val_df.index)
    test_marks = time_features(test_df.index)
    
    if config.PREDICT_RETURNS:
        scaler_cov = StandardScaler()
        scaler_target = StandardScaler()
    else:
        scaler_cov = MinMaxScaler()
        scaler_target = MinMaxScaler()
    
    train_cov = scaler_cov.fit_transform(train_vals[:, 1:])
    val_cov = scaler_cov.transform(val_vals[:, 1:])
    test_cov = scaler_cov.transform(test_vals[:, 1:])
    
    train_target = scaler_target.fit_transform(train_vals[:, 0:1])
    val_target = scaler_target.transform(val_vals[:, 0:1])
    test_target = scaler_target.transform(test_vals[:, 0:1])
    
    train_vals = np.hstack([train_target, train_cov])
    val_vals = np.hstack([val_target, val_cov])
    test_vals = np.hstack([test_target, test_cov])
    
    input_dim = len(feature_cols)
    print(f"    Features in scenario: {input_dim}")
    
    return (train_vals, train_marks, 
            val_vals, val_marks, 
            test_vals, test_marks, 
            input_dim, scaler_target)

# ==============================
# DATASET CLASS (MATCHING INFORMER EXPECTATIONS)
# ==============================

class InformerDataset(Dataset):
    def __init__(self, data, dates, seq_len, label_len, pred_len):
        self.data = data
        self.dates = dates
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        seq_x_mark = self.dates[s_begin:s_end]
        seq_y_mark = self.dates[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32), 
            torch.tensor(seq_y_mark, dtype=torch.float32),
            torch.tensor(self.data[s_end:s_end + self.pred_len, 0:1], dtype=torch.float32) # Target
        )

# ==============================
# MODEL & TRAINING
# ==============================

class DirectionalMSELoss(nn.Module):
    def __init__(self, direction_weight: float = 1.5, mse_weight: float = 1.0):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        if pred.size(1) > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            wrong_direction = (torch.sign(pred_diff) != torch.sign(target_diff)).float()
            direction_penalty = torch.mean(wrong_direction)
        else:
            wrong_direction = (torch.sign(pred) != torch.sign(target)).float()
            direction_penalty = torch.mean(wrong_direction)
        return self.mse_weight * mse_loss + self.direction_weight * direction_penalty

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

def train_torch_model(model: nn.Module, train_loader: DataLoader, 
                     val_loader: DataLoader, epochs: int, lr: float, 
                     patience: int, model_save_path: str) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    if config.USE_DIRECTIONAL_LOSS:
        criterion = DirectionalMSELoss(
            direction_weight=config.DIRECTION_LOSS_WEIGHT,
            mse_weight=config.MSE_LOSS_WEIGHT
        )
    else:
        criterion = nn.MSELoss()
    
    early_stopper = EarlyStopping(patience=patience)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            enc, enc_mark, dec, dec_mark, y = [b.to(config.DEVICE) for b in batch]
            
            # Mask decoder input
            dec_input = dec.clone()
            dec_input[:, -config.PREDICTION_HORIZONS[0]:, 0] = 0 # zero out target

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Call Informer (Note: training_target argument is optional in Informer)
                pred = model(enc, enc_mark, dec_input, dec_mark)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                enc, enc_mark, dec, dec_mark, y = [b.to(config.DEVICE) for b in batch]
                dec_input = dec.clone()
                dec_input[:, -config.PREDICTION_HORIZONS[0]:, 0] = 0
                
                pred = model(enc, enc_mark, dec_input, dec_mark)
                val_loss += criterion(pred, y).item()
        
        avg_val = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val)
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), model_save_path)
        
        if early_stopper(avg_val):
            break

    model.load_state_dict(torch.load(model_save_path))
    return best_loss

def create_ensemble_models(input_dim: int, n_models: int = 3) -> List[nn.Module]:
    models = []
    
    horizon = config.PREDICTION_HORIZONS[0]
    lookback = config.LOOKBACK_WINDOWS[0]
    label_len = lookback // 2
    
    for i in range(n_models):
        model_dropout = min(config.DROPOUT + (i * 0.02), 0.5)
        
        # Instantiate INFORMER with specific config
        model = Informer(
            input_dim=input_dim,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            out_len=horizon,
            distil=config.DISTIL,        # False
            prob_sparse=config.PROB_SPARSE, # False
            dec_only=config.DEC_ONLY,    # False
            gen_dec=config.GEN_DEC,      # True
            dropout=model_dropout
        ).to(config.DEVICE)
        
        models.append(model)
    return models

def ensemble_predict(models: List[nn.Module], data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    all_preds = []
    all_actuals = None
    
    horizon = config.PREDICTION_HORIZONS[0]
    
    for model in models:
        model.eval()
        model_preds = []
        model_actuals = []
        with torch.no_grad():
            for batch in data_loader:
                enc, enc_mark, dec, dec_mark, y = [b.to(config.DEVICE) for b in batch]
                
                dec_input = dec.clone()
                dec_input[:, -horizon:, 0] = 0
                
                pred = model(enc, enc_mark, dec_input, dec_mark)
                model_preds.append(pred.cpu().numpy())
                model_actuals.append(y.cpu().numpy())
        
        all_preds.append(np.concatenate(model_preds))
        if all_actuals is None:
            all_actuals = np.concatenate(model_actuals)
            
    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds, all_actuals

# ==============================
# MAIN ABLATION LOOP
# ==============================

def run_ablation_study():
    results = []
    horizon = config.PREDICTION_HORIZONS[0]
    lookback = config.LOOKBACK_WINDOWS[0]
    label_len = lookback // 2
    
    print(f"\\nSTARTING ABLATION STUDY for Horizon {horizon}d")
    print(f"Lookback: {lookback}, Batch: {config.BATCH_SIZE}")
    
    for scenario_name, _ in ABLATION_SCENARIOS.items():
        print(f"\\n{'='*50}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*50}")
        
        # 1. Prepare Data
        cols = get_feature_columns(scenario_name)
        try:
            data = load_and_split_data(target_col='Close', feature_columns=cols)
            train_vals, train_marks, val_vals, val_marks, test_vals, test_marks, input_dim, scaler_target = data
        except Exception as e:
            print(f"Skipping scenario {scenario_name} due to data error: {e}")
            continue
            
        # 2. Prepare Loaders
        train_ds = InformerDataset(train_vals, train_marks, lookback, label_len, horizon)
        val_ds = InformerDataset(val_vals, val_marks, lookback, label_len, horizon)
        test_ds = InformerDataset(test_vals, test_marks, lookback, label_len, horizon)
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # 3. Create & Train Ensemble
        models = create_ensemble_models(input_dim, n_models=config.N_ENSEMBLE_MODELS)
        
        ensemble_losses = []
        for i, model in enumerate(models):
            print(f"  Training Model {i+1}...")
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"{scenario_name}_m{i}.pth")
            loss = train_torch_model(model, train_loader, val_loader, config.EPOCHS, 1e-4, config.PATIENCE, save_path)
            ensemble_losses.append(loss)
            
        print(f"  Ensemble val loss: {np.mean(ensemble_losses):.6f}")
        
        # 4. Predict & Evaluate
        preds_scaled, actuals_scaled = ensemble_predict(models, test_loader)
        
        preds_actual = scaler_target.inverse_transform(preds_scaled.reshape(-1, 1))
        actuals_actual = scaler_target.inverse_transform(actuals_scaled.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(actuals_actual, preds_actual))
        mae = mean_absolute_error(actuals_actual, preds_actual)
        
        print(f"  > Test RMSE: {rmse:.4f}")
        print(f"  > Test MAE:  {mae:.4f}")
        
        # 5. Save Results
        res = {
            "Scenario": scenario_name,
            "Horizon": horizon,
            "RMSE": rmse,
            "MAE": mae,
            "Features": input_dim,
            "Timestamp": datetime.now().isoformat()
        }
        results.append(res)
        append_result_to_csv(res)

    print("\\n\\nABLATION STUDY COMPLETED!")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run_ablation_study()