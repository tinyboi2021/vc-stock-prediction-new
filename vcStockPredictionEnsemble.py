# ==========================================
# 0. SAFE DEPENDENCY INSTALLATION (Background Safe)
# ==========================================
# Install general tools normally
!pip install optuna openpyxl mlflow seaborn dagshub --quiet
# Install Captum WITHOUT dependencies to prevent it from downgrading NumPy!
!pip install captum --no-deps --quiet

# ==========================================
# 1. STANDARD IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.utils import weight_norm
from sklearn.mixture import GaussianMixture
from IPython.display import display, HTML
import math
import os
import warnings
import optuna
import random
import copy
import sys
import re
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
from kaggle_secrets import UserSecretsClient

# 1. Set explicit MLflow environment variables for DagsHub
try:
    user_secrets = UserSecretsClient()
    dagshub_token = user_secrets.get_secret("DAGSHUB_TOKEN") 
except:
    # Fallback to hardcoded if secrets aren't configured yet (Not recommended for sharing!)
    dagshub_token = ""

os.environ["MLFLOW_TRACKING_USERNAME"] = "tomtony123456789"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RogueNinja240/vc-stock-prediction.mlflow"

# 2. Tell MLflow exactly where to point
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# 3. Explicitly set an experiment name so runs are grouped cleanly
mlflow.set_experiment("NextDaySentimentPeak")

print(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

# Note: We completely removed dagshub.init() because setting os.environ is much more reliable!

torch.set_float32_matmul_precision('medium')
# Enable TF32 for A100/H100 optimization
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# ==========================================
# 🎯 TEAM ASSIGNMENTS (CHANGE THESE BEFORE RUNNING)
# ==========================================
# Paul S: SCENARIO = "With Sentiment"    | HORIZON = 1
# Tom T: SCENARIO = "With Sentiment"    | HORIZON = 8
# Hareesh S: SCENARIO = "With Sentiment"    | HORIZON = 16
# Paul S: SCENARIO = "Without Sentiment" | HORIZON = 1
# Tom T: SCENARIO = "Without Sentiment" | HORIZON = 8
# Hareesh S: SCENARIO = "Without Sentiment" | HORIZON = 16

TARGET_SCENARIO = "With Sentiment"  # Options: "With Sentiment" OR "Without Sentiment"
TARGET_HORIZON = 8       # Options: 1, 8, or 16
RUNNER_NAME = "Tom T"            # Put your name here for MLflow tracking!

# ==========================================
# 1. CONFIGURATION & SEEDING
# ==========================================

# NOTE: Ensure this path is correct for your environment
DATA_PATH = "/kaggle/input/datasets/tomtonythegreat/finalpipelinedataset/final_sentiment_scored_dataset.csv" 
OUTPUT_FILE = f"./Results_{TARGET_SCENARIO.replace(' ', '_')}_H{TARGET_HORIZON}.xlsx"
MODEL_SAVE_DIR = "./Saved_Models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
PRED_SAVE_DIR = "./Prediction_Excel_Sheets"
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
USE_WALK_FORWARD = False
ENABLE_REVIN = True          # ⚙️ Set to False to bypass RevIN normalization
ENABLE_CUSTOM_LOSS = True   # ⚙️ Set to False to force pure MSE loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device detected: {DEVICE}")

EPOCHS = 300 # Reduced for demonstration; set to 300 for full paper results
PATIENCE = 20
N_TRIALS = 100 # Reduced for demonstration; set to 50 for full optimization

# Create global tracking dict for MLflow
MLFLOW_TAGS = {"Runner": RUNNER_NAME, "Track": f"{TARGET_SCENARIO}_H{TARGET_HORIZON}"}

# --- NEW: Checkpoint & Extended Excel Configurations ---
CHECKPOINT_FILE = "./execution_checkpoint.csv"
SUMMARY_EXCEL_FILE = f"./Best_Models_Summary_{TARGET_SCENARIO.replace(' ', '_')}_H{TARGET_HORIZON}.xlsx"
XAI_EXCEL_FILE = f"./XAI_Results_{TARGET_SCENARIO.replace(' ', '_')}_H{TARGET_HORIZON}.xlsx"

# Enable system metrics logging (CPU, RAM, GPU) in MLflow if supported
try:
    mlflow.enable_system_metrics_logging()
except Exception:
    pass

def is_completed(scenario, lookback, horizon, model_name):
    """Checks if a specific model run has already completed successfully."""
    if not os.path.exists(CHECKPOINT_FILE):
        return False
    df = pd.read_csv(CHECKPOINT_FILE)
    match = df[(df['Scenario'] == scenario) & 
               (df['Lookback'] == lookback) & 
               (df['Horizon'] == horizon) & 
               (df['Model'] == model_name)]
    return not match.empty

def mark_completed(scenario, lookback, horizon, model_name):
    """Marks a run as completed in the checkpoint CSV."""
    record = {"Scenario": scenario, "Lookback": lookback, "Horizon": horizon, "Model": model_name}
    df = pd.DataFrame([record])
    if not os.path.exists(CHECKPOINT_FILE):
        df.to_csv(CHECKPOINT_FILE, index=False)
    else:
        df.to_csv(CHECKPOINT_FILE, mode='a', header=False, index=False)

# ==========================================
# 0. ELEGANT LOGGING CONFIGURATION
# ==========================================
def cprint(msg):
    """Drop-in replacement for print() to render HTML formatted strings."""
    # Strip HTML tags for the standard log output
    clean_msg = re.sub('<[^<]+>', '', msg)
    clean_msg = clean_msg.replace('&nbsp;', ' ')
    print(clean_msg) 
    # Still send to display for interactive notebook viewing
    display(HTML(msg))
class ColorLog:
    # Replace ANSI codes with HTML tags
    HEADER = '<span style="color: #FF00FF;">'  # Magenta
    OKBLUE = '<span style="color: #0000FF;">'  # Blue
    OKCYAN = '<span style="color: #00CED1;">'  # Cyan
    OKGREEN = '<span style="color: #008000;">' # Green
    WARNING = '<span style="color: #FFA500;">' # Orange
    FAIL = '<span style="color: #FF0000;">'    # Red
    ENDC = '</span>'
    BOLD = '<b>'
    UNDERLINE = '<u>'

    @staticmethod
    def info(msg): 
        display(HTML(f"{ColorLog.OKBLUE}<b>[INFO]</b>{ColorLog.ENDC} {msg}"))
        
    @staticmethod
    def success(msg): 
        display(HTML(f"{ColorLog.OKGREEN}<b>[SUCCESS]</b>{ColorLog.ENDC} {msg}"))
        
    @staticmethod
    def warn(msg): 
        display(HTML(f"{ColorLog.WARNING}<b>[WARN]</b>{ColorLog.ENDC} {msg}"))
        
    @staticmethod
    def section(msg):
        # Use &nbsp; to preserve the centering spaces in HTML
        centered_msg = msg.center(60).replace(' ', '&nbsp;')
        html_str = f"""
        <br>
        {ColorLog.BOLD}{ColorLog.HEADER}{'='*60}{ColorLog.ENDC} <br>
        {ColorLog.BOLD}{ColorLog.HEADER}{centered_msg}{ColorLog.ENDC} <br>
        {ColorLog.BOLD}{ColorLog.HEADER}{'='*60}{ColorLog.ENDC}
        """
        display(HTML(html_str))
    @staticmethod
    def extreme(msg, color="#00FF00", border_color="#FF00FF"):
        """Creates a massive, highly visible banner for major milestones."""
        
        # 1. Standard print fallback for Kaggle console and background logs
        print(f"\n{'='*60}")
        print(f"🚨 >>> {msg.upper()} <<< 🚨".center(60))
        print(f"{'='*60}\n")
        
        # 2. Keep the HTML execution for when you are viewing interactively
        html_str = f"""
        <div style="background-color: #1e1e1e; color: {color}; padding: 15px; 
                    border: 4px dashed {border_color}; border-radius: 8px; 
                    text-align: center; font-size: 20px; font-weight: 900; 
                    margin: 25px 0; text-transform: uppercase; font-family: monospace;
                    box-shadow: 0px 0px 15px {border_color};">
            🚨 >>> {msg} <<< 🚨
        </div>
        """
        display(HTML(html_str))
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    ColorLog.info(f"Seed set to {seed}. Deterministic algorithms enabled.")
class DeepLearningVisualizer:
    """
    A centralized module for generating high-fidelity plots and logging them 
    directly to MLflow as artifacts to ensure team-wide visibility.
    """
    
    # Define a clean, professional aesthetic for all plots
    plt.style.use('dark_background') 
    PLOT_DIR = "./temp_mlflow_plots"

    @classmethod
    def _ensure_dir(cls):
        os.makedirs(cls.PLOT_DIR, exist_ok=True)
    @classmethod
    def generate_readme(cls, folder_name, content):
        """Generates a README text file and logs it to MLflow."""
        cls._ensure_dir()
        
        # 1. Construct the full path to the file
        readme_path = os.path.join(cls.PLOT_DIR, f"README_{folder_name}.txt")
        
        # 2. Extract the directory part of the path and ensure it exists
        readme_dir = os.path.dirname(readme_path)
        os.makedirs(readme_dir, exist_ok=True)
        
        # 3. Now it is safe to open and write to the file
        with open(readme_path, "w") as f:
            f.write(content)
            
        if mlflow.active_run():
            mlflow.log_artifact(readme_path, artifact_path=folder_name)
            
        if os.path.exists(readme_path): os.remove(readme_path)

    @classmethod
    def plot_and_log_global_trajectory(cls, all_actuals, all_preds, model_name, horizon, lookback):
        cls.generate_readme("global_trajectories", "GLOBAL TRAJECTORY PLOTS\n\nThese plots stitch together the first step of every prediction window to show continuous performance over the entire test set. \n- Blue Line: Actual Stock Price\n- Orange Line: Model Forecast\nLook for areas where the orange line significantly lags or overshoots the blue line to identify regime-specific weaknesses.")
        """
        Stitches together the first step of every prediction window to show 
        the continuous global performance over the entire test set.
        """
        cls._ensure_dir()
        fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
        
        # We take the first predicted step (h=0) from every window to create a continuous line
        global_act = all_actuals[:, 0, 0]
        global_pred = all_preds[:, 0, 0]
        
        ax.plot(global_act, label='Actual Price', color='#00CED1', alpha=0.8, linewidth=1.5)
        ax.plot(global_pred, label=f'{model_name} Forecast', color='#FFA500', alpha=0.9, linewidth=1.5)
        
        ax.set_title(f'Global Test Set Trajectory: {model_name}\n(Lookback: {lookback} | Horizon: {horizon})', fontsize=14, pad=15)
        ax.set_xlabel('Time (Test Set Steps)', fontsize=12)
        ax.set_ylabel('Price (Scaled)', fontsize=12)
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plot_path = os.path.join(cls.PLOT_DIR, f"{model_name}_Global_Trajectory_L{lookback}_H{horizon}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, artifact_path=f"global_trajectories")
        if os.path.exists(plot_path): os.remove(plot_path)

    @classmethod
    def plot_and_log_error_distribution(cls, actuals, predictions, model_name, horizon):
        """
        Plots a histogram/KDE of the prediction errors to check for bias and outliers.
        """
        cls._ensure_dir()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # Flatten arrays and calculate pure error
        errors = actuals.flatten() - predictions.flatten()
        
        sns.histplot(errors, bins=50, kde=True, color='#FF6347', edgecolor='white', ax=ax)
        ax.axvline(x=0, color='white', linestyle='--', linewidth=2, label='Zero Error')
        
        ax.set_title(f'Error Distribution (Actual - Predicted): {model_name}\n(Horizon: {horizon})', fontsize=14, pad=15)
        ax.set_xlabel('Error Magnitude (Scaled)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plot_path = os.path.join(cls.PLOT_DIR, f"{model_name}_Error_Dist_H{horizon}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, artifact_path=f"error_distributions")
        if os.path.exists(plot_path): os.remove(plot_path)

    @classmethod
    def plot_and_log_uncertainty_fan(cls, actual_seq, pred_mean_seq, lower_bound, upper_bound, model_name, horizon):
        """
        Plots a specific forecast trajectory with a shaded 95% confidence interval.
        """
        cls._ensure_dir()
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        time_steps = np.arange(len(actual_seq))
        
        ax.plot(time_steps, actual_seq, label='Actual Price', color='#00CED1', marker='o')
        ax.plot(time_steps, pred_mean_seq, label='Mean Forecast', color='#FFA500', linestyle='--')
        
        # Shade the area between the upper and lower bounds
        ax.fill_between(time_steps, lower_bound, upper_bound, color='#FFA500', alpha=0.2, label='95% Confidence Interval')
        
        ax.set_title(f'Forecast Uncertainty (MC-Dropout): {model_name}\n(Horizon: {horizon})', fontsize=14, pad=15)
        ax.set_xlabel('Days into Future', fontsize=12)
        ax.set_ylabel('Price (Scaled)', fontsize=12)
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plot_path = os.path.join(cls.PLOT_DIR, f"{model_name}_Uncertainty_H{horizon}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, artifact_path=f"uncertainty_plots")
        if os.path.exists(plot_path): os.remove(plot_path)
    @classmethod
    def plot_and_log_forecast(cls, actuals, predictions, model_name, horizon, lookback, picp=None):
        """
        Generates a line chart comparing the predicted trajectory against the actual ground truth.
        """
        cls._ensure_dir()
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        # Plot the first sequence in the test batch for a clear visual comparison
        # Shape is expected to be [Batch, Horizon, 1]. We flatten it for 1D plotting.
        act_seq = actuals[0, :, 0]
        pred_seq = predictions[0, :, 0]
        
        ax.plot(act_seq, label='Actual Price', color='#00CED1', marker='o', linewidth=2)
        ax.plot(pred_seq, label=f'{model_name} Forecast', color='#FFA500', linestyle='--', marker='x', linewidth=2)
        
        title = f'AAPL Forecast Trajectory: {model_name}\n(Lookback: {lookback} | Horizon: {horizon} Days)'
        if picp is not None:
            title += f' | PICP (Confidence): {picp:.2%}'
            
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('Days into Future', fontsize=12)
        ax.set_ylabel('Price (Scaled)', fontsize=12)
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plot_path = os.path.join(cls.PLOT_DIR, f"{model_name}_forecast_L{lookback}_H{horizon}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        # Log and cleanup
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, artifact_path=f"forecasts/H_{horizon}")
        if os.path.exists(plot_path): os.remove(plot_path)

    @classmethod
    def plot_and_log_xai_heatmap(cls, npy_filepath, feature_cols, model_name, horizon):
        cls.generate_readme(f"xai_heatmaps/H_{horizon}", "XAI FEATURE IMPORTANCE HEATMAPS\n\nThese heatmaps show which historical features drove the model's decisions. \n- Y-Axis: Time lags (historical steps). Top is most recent, bottom is furthest in the past.\n- X-Axis: Input features (e.g., Close, RSI, Sentiment).\n- Color Intensity: Brighter colors (yellow/white) mean higher importance for that specific feature at that specific time lag.")
        """
        Reads the 3D XAI Integrated Gradients array from disk and generates a 
        2D heatmap showing which features drove the model's decisions over time.
        """
        if not os.path.exists(npy_filepath):
            print(f"Warning: XAI artifact {npy_filepath} not found for plotting.")
            return

        cls._ensure_dir()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # Load the array: Expected shape [Horizon, SeqLen, Features]
        temporal_attr = np.load(npy_filepath)
        
        # Aggregate the attribution strength across the prediction horizon to get a 2D map [SeqLen, Features]
        # Using absolute values to show magnitude of importance regardless of direction
        heatmap_data = np.abs(temporal_attr).mean(axis=0)
        
        sns.heatmap(heatmap_data, 
                    cmap='magma', 
                    xticklabels=feature_cols, 
                    yticklabels=False, # Hide y-ticks to keep it clean, represents time lags
                    cbar_kws={'label': 'Gradient Attribution Magnitude'},
                    ax=ax)
        
        ax.set_title(f'XAI Feature Importance Heatmap: {model_name}\n(Averaged across {horizon}-Day Horizon)', fontsize=14, pad=15)
        ax.set_xlabel('Input Features', fontsize=12)
        ax.set_ylabel('Historical Sequence Length (Lag)', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        
        plot_path = os.path.join(cls.PLOT_DIR, f"{model_name}_XAI_Heatmap_H{horizon}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, artifact_path=f"xai_heatmaps/H_{horizon}")
        if os.path.exists(plot_path): os.remove(plot_path)

    @classmethod
    def plot_and_log_ensemble_weights(cls, model_names, weights, ensemble_name, horizon, lookback):
        """
        Creates a horizontal bar chart showing the Inverse-MSE weights assigned 
        to each model in the dynamic ensemble.
        """
        cls._ensure_dir()
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        # Sort weights for better visual hierarchy
        sorted_indices = np.argsort(weights)
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0.4, 0.9, len(weights)))
        
        bars = ax.barh(sorted_names, sorted_weights, color=colors)
        
        ax.set_title(f'Dynamic Ensemble Weight Distribution\n({ensemble_name} | Lookback: {lookback} | Horizon: {horizon})', fontsize=14, pad=15)
        ax.set_xlabel('Assigned Weight (%)', fontsize=12)
        
        # Add exact percentages as text on the bars
        for bar in bars:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.1%}', 
                    va='center', ha='left', color='white', fontweight='bold')
            
        ax.set_xlim(0, max(weights) + 0.1) # Add padding for the text
        ax.grid(True, axis='x', alpha=0.2, linestyle=':')
        
        plot_path = os.path.join(cls.PLOT_DIR, f"{ensemble_name}_Weights_L{lookback}_H{horizon}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        if mlflow.active_run():
            mlflow.log_artifact(plot_path, artifact_path="ensemble_weights")
        if os.path.exists(plot_path): os.remove(plot_path)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
def pretrain_patchtst_model(model, train_loader, val_loader, epochs, lr, patience, model_save_path):
    """
    Self-supervised pretraining loop for PatchTST using Masked Patch Reconstruction.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss(reduction='none') # No reduction so we can apply the boolean mask

    best_loss = float('inf')
    early_stopper = EarlyStopping(patience=patience) if val_loader is not None else None

    ColorLog.extreme("STARTING SELF-SUPERVISED PRETRAINING", color="#8A2BE2", border_color="#8A2BE2")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = [b.to(DEVICE) for b in batch]
            enc = batch[0] # Target series sequence

            # Forward pass in pretrain mode
            rec_patches, patches_orig, mask = model(enc, pretrain=True, mask_ratio=0.4)

            # Expand mask [B, C, Num_Patches] -> [B, C, Num_Patches, Patch_Len]
            mask_expanded = mask.unsqueeze(-1).expand_as(rec_patches)

            # Compute loss ONLY on the masked patches
            loss_matrix = criterion(rec_patches, patches_orig)
            loss = (loss_matrix * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

            loss.backward()
            # 5️⃣ Gradient Monitoring: Capture the grad norm returned by clip_grad_norm_
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            # Log grad norm to MLflow if a run is active (useful during final training phase)
            if mlflow.active_run():
                mlflow.log_metric("grad_norm", grad_norm.item(), step=epoch)

        avg_train_loss = train_loss / len(train_loader)
        
        # Save weights
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), model_save_path)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        
    return best_loss
def save_prediction_details(scenario, lookback, horizon, model_name, 
                            actuals_scaled, preds_scaled, scaler_target):
    ColorLog.extreme(f"EXPORTING PREDICTIONS TO EXCEL: {model_name}", color="#32CD32", border_color="#32CD32")
    """
    Inverse-transforms predictions and saves them to a model-specific sheet
    within an Excel file dedicated to the Scenario/Horizon/Lookback combo.
    """
    # 1. Inverse Transform to Real Prices
    # actuals_scaled/preds_scaled shape: (N, Horizon, 1)
    N, H, _ = actuals_scaled.shape
    
    actuals_flat = actuals_scaled.reshape(-1, 1)
    preds_flat = preds_scaled.reshape(-1, 1)
    
    actuals_real = scaler_target.inverse_transform(actuals_flat).reshape(N, H)
    preds_real = scaler_target.inverse_transform(preds_flat).reshape(N, H)
    
    # 2. Create DataFrames
    # For multi-step (Horizon > 1), we create columns: Step_1, Step_2...
    df_data = {}
    for h_idx in range(H):
        df_data[f"Actual_Step_{h_idx+1}"] = actuals_real[:, h_idx]
        df_data[f"Predicted_Step_{h_idx+1}"] = preds_real[:, h_idx]
        df_data[f"Error_Step_{h_idx+1}"] = actuals_real[:, h_idx] - preds_real[:, h_idx]
    
    df = pd.DataFrame(df_data)
    
    # 3. Construct File and Sheet Name
    clean_scenario = scenario.replace(" ", "_")
    file_name = f"Predictions_{clean_scenario}_L{lookback}_H{horizon}.xlsx"
    file_path = os.path.join(PRED_SAVE_DIR, file_name)
    
    # 4. Save using ExcelWriter (Append mode if file exists)
    mode = 'a' if os.path.exists(file_path) else 'w'
    if mode == 'a':
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=model_name[:31], index=False) # Excel limit is 31 chars
    else:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=model_name[:31], index=False)
# ==========================================
# 3. MODEL ARCHITECTURES (WITH REVIN & FULL ATTENTION)
# ==========================================

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, enable_revin=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.enable_revin = enable_revin # ⚙️ NEW TOGGLE
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _normalize(self, x, mean=None, stdev=None, gamma=None, beta=None):
        if not self.enable_revin:
            # Bypass normalization entirely
            dummy_mean = torch.zeros_like(x[:, 0:1, :])
            dummy_stdev = torch.ones_like(x[:, 0:1, :])
            return x, dummy_mean, dummy_stdev, None, None

        if mean is None or stdev is None:
            dim2reduce = tuple(range(1, x.ndim - 1))
            mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
            var = torch.clamp(var, min=1e-5)
            stdev = torch.sqrt(var).detach()
        
        x = x - mean
        x = x / (stdev + self.eps)
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x, mean, stdev, None, None

    def _denormalize(self, x, mean, stdev, gamma=None, beta=None, target_idx=None):
        if not self.enable_revin:
            # Bypass denormalization entirely
            return x

        if self.affine:
            if target_idx is not None:
                x = x - self.affine_bias[target_idx]
                x = x / (torch.abs(self.affine_weight[target_idx]) + self.eps)
            else:
                x = x - self.affine_bias
                x = x / (torch.abs(self.affine_weight) + self.eps)

        if target_idx is not None:
            target_mean = mean[..., target_idx:target_idx+1]
            target_stdev = stdev[..., target_idx:target_idx+1]
            x = x * target_stdev
            x = x + target_mean
        else:
            x = x * stdev
            x = x + mean
        return x

    def forward(self, x, mode: str, mean=None, stdev=None, gamma=None, beta=None, target_idx=None):
        if mode == 'norm':
            return self._normalize(x, mean, stdev, gamma, beta)
        elif mode == 'denorm':
            return self._denormalize(x, mean, stdev, gamma, beta, target_idx)


class SANorm(nn.Module):
    def __init__(self, num_features: int, seq_len: int, num_slices: int = 4, eps=1e-5, affine=True):
        super(SANorm, self).__init__()
        self.num_features = num_features
        self.num_slices = num_slices
        self.slice_len = max(1, seq_len // num_slices)
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_net = nn.Sequential(
                nn.Linear(num_features, num_features // 2),
                nn.ReLU(),
                nn.Linear(num_features // 2, num_features * 2)
            )

    def _normalize(self, x, mean=None, stdev=None, gamma=None, beta=None):
        B, L, C = x.shape
        
        if mean is not None and stdev is not None:
            x_norm = (x - mean) / stdev
            if self.affine and gamma is not None and beta is not None:
                x_norm = x_norm * gamma + beta
            return x_norm, mean, stdev, gamma, beta

        pad_len = (self.num_slices - L % self.num_slices) % self.num_slices
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
        else:
            x_padded = x
            
        x_slices = x_padded.view(B, self.num_slices, -1, C)
        slice_mean = torch.mean(x_slices, dim=2, keepdim=True)
        slice_var = torch.var(x_slices, dim=2, keepdim=True, unbiased=False)
        slice_stdev = torch.sqrt(torch.clamp(slice_var, min=self.eps))
        
        x_norm = (x_slices - slice_mean) / slice_stdev
        
        out_gamma, out_beta = None, None
        if self.affine:
            slice_stats = torch.mean(x_slices, dim=2) 
            affine_params = self.affine_net(slice_stats).unsqueeze(2) 
            g, b = torch.chunk(affine_params, 2, dim=-1)
            x_norm = x_norm * g + b
            
            # ⚙️ CRITICAL FIX: Return parameters dynamically instead of storing
            out_gamma = g[:, -1, :, :]
            out_beta = b[:, -1, :, :]
            
        x_norm = x_norm.view(B, -1, C)[:, :L, :]
        
        last_mean = slice_mean[:, -1, :].detach()
        last_stdev = slice_stdev[:, -1, :].detach()
        return x_norm, last_mean, last_stdev, out_gamma, out_beta

    def _denormalize(self, x, mean, stdev, gamma=None, beta=None, target_idx=None):
        if self.affine and gamma is not None and beta is not None:
            g = gamma
            b = beta
            
            if target_idx is not None:
                g = g[..., target_idx:target_idx+1]
                b = b[..., target_idx:target_idx+1]

            x = (x - b) / (g + self.eps)

        if target_idx is not None:
            mean = mean[..., target_idx:target_idx+1]
            stdev = stdev[..., target_idx:target_idx+1]
            
        return (x * stdev) + mean

    def forward(self, x, mode: str, mean=None, stdev=None, gamma=None, beta=None, target_idx=None):
        if mode == 'norm':
            return self._normalize(x, mean, stdev, gamma, beta)
        elif mode == 'denorm':
            return self._denormalize(x, mean, stdev, gamma, beta, target_idx)


class WDANorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-5):
        super(WDANorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        self.trend_gamma = nn.Parameter(torch.ones(num_features))
        self.trend_beta = nn.Parameter(torch.zeros(num_features))
        self.fluc_gamma = nn.Parameter(torch.ones(num_features))
        self.fluc_beta = nn.Parameter(torch.zeros(num_features))

    def _normalize(self, x, mean=None, stdev=None, gamma=None, beta=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if mean is None or stdev is None:
            mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
            stdev = torch.sqrt(torch.clamp(var, min=self.eps)).detach()
        
        # ⚙️ FIX: Use clone to protect computational graph
        x_norm = (x.clone() - mean) / stdev

        padding = 4 // 2
        trend = F.avg_pool1d(x_norm.permute(0, 2, 1), kernel_size=5, stride=1, padding=padding).permute(0, 2, 1)
        fluc = x_norm - trend
        
        trend_out = trend * self.trend_gamma + self.trend_beta
        fluc_out = fluc * self.fluc_gamma + self.fluc_beta
        
        out = trend_out + fluc_out
        
        return out, mean, stdev, None, None

    def _denormalize(self, x, mean, stdev, gamma=None, beta=None, target_idx=None):
        if target_idx is not None:
            x = (x * stdev[..., target_idx:target_idx+1]) + mean[..., target_idx:target_idx+1]
        else:
            x = (x * stdev) + mean
        return x

    def forward(self, x, mode: str, mean=None, stdev=None, gamma=None, beta=None, target_idx=None):
        if mode == 'norm':
            return self._normalize(x, mean, stdev, gamma, beta)
        elif mode == 'denorm':
            return self._denormalize(x, mean, stdev, gamma, beta, target_idx)


class CoRANorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-5):
        super(CoRANorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        if num_features > 1:
            self.covariate_attention = nn.Sequential(
                nn.Linear(num_features - 1, 16),
                nn.ReLU(),
                nn.Linear(16, 2) 
            )

    def _normalize(self, x, mean=None, stdev=None, gamma=None, beta=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if mean is None or stdev is None:
            mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
            stdev = torch.sqrt(torch.clamp(var, min=self.eps)).detach()
        
        # ⚙️ FIX: Clone the tensor before performing inplace operations on target dimension
        x_norm = (x.clone() - mean) / stdev
        
        out_gamma, out_beta = None, None
        if self.num_features > 1:
            if gamma is not None and beta is not None:
                t_gamma, t_beta = gamma, beta
                out_gamma, out_beta = gamma, beta
            else:
                covariates_mean = mean[..., 1:].squeeze(1) 
                target_shift = self.covariate_attention(covariates_mean).unsqueeze(1) 
                t_gamma, t_beta = torch.chunk(target_shift, 2, dim=-1)
                out_gamma, out_beta = t_gamma, t_beta
            
            # Avoid inplace modifications
            target_feat = x_norm[..., 0:1] * t_gamma + t_beta
            covar_feats = x_norm[..., 1:]
            x_norm = torch.cat([target_feat, covar_feats], dim=-1)
            
        return x_norm, mean, stdev, out_gamma, out_beta

    def _denormalize(self, x, mean, stdev, gamma=None, beta=None, target_idx=None):
        if self.num_features > 1 and gamma is not None and beta is not None:
            if target_idx == 0 or target_idx is None:
                # Target feature denormalization
                x_target = (x[..., 0:1] - beta) / (gamma + self.eps)
                if target_idx is None:
                    x = torch.cat([x_target, x[..., 1:]], dim=-1)
                else:
                    x = x_target

        if target_idx is not None:
            x = x * stdev[..., target_idx:target_idx+1]
            x = x + mean[..., target_idx:target_idx+1]
        else:
            x = x * stdev + mean
        return x

    def forward(self, x, mode: str, mean=None, stdev=None, gamma=None, beta=None, target_idx=None):
        if mode == 'norm':
            return self._normalize(x, mean, stdev, gamma, beta)
        elif mode == 'denorm':
            return self._denormalize(x, mean, stdev, gamma, beta, target_idx)
class TCNModel(nn.Module):
    # ⚙️ MINIMAL CHANGE: Added enable_revin=True to the parameters
    def __init__(self, input_dim, output_len=1, num_channels=[64, 64], kernel_size=2, dropout=0.2, enable_revin=True):
        super(TCNModel, self).__init__()
        
        # ⚙️ MINIMAL CHANGE: Passed the toggle to RevIN
        self.revin = RevIN(input_dim, affine=True, enable_revin=enable_revin)
        
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_ch, num_channels[i], kernel_size, 1, dilation_size, padding, dropout)]
        self.tcn = nn.Sequential(*layers)
        self.projection = nn.Linear(num_channels[-1], output_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, training_target=None):
        # 1. Normalization 
        x, mean, stdev, gamma, beta = self.revin(x_enc, 'norm')
        
        # 2. TCN Forward Pass
        x = x.permute(0, 2, 1) 
        out = self.tcn(x)[:, :, -1] 
        
        # 3. Projection to Horizon
        pred = self.projection(out).unsqueeze(-1)
        
        # 4. MINIMAL CHANGE: Restored internal Denormalization
        # This returns the prediction back to the original MinMaxScaler (0-1) scale.
        return self.revin(pred, 'denorm', mean=mean, stdev=stdev, gamma=gamma, beta=beta, target_idx=0)

def save_results_to_excel(final_data, filename):
    try:
        df_final = pd.DataFrame(final_data)
        if os.path.exists(filename):
            # Read existing to append
            df_existing = pd.read_excel(filename)
            df_final = pd.concat([df_existing, df_final], ignore_index=True)
            df_final.to_excel(filename, index=False)
        else:
            df_final.to_excel(filename, index=False)
        print(f"--- Results updated and saved to {filename} ---")
    except Exception as e:
        print(f"Warning: Could not save results. Error: {e}")

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================

def time_features(dates):
    return np.vstack([
        dates.month,
        dates.day,
        dates.weekday,
        dates.hour
    ]).transpose()

class InformerDataset(Dataset):
    def __init__(self, data, dates, regimes, seq_len, label_len, pred_len, 
                 sent_col_idx=None, adx_col_idx=None):
        self.data = data
        self.dates = dates
        self.regimes = regimes # 🚀 NEW: Store GMM Regimes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.sent_col_idx = sent_col_idx
        self.adx_col_idx = adx_col_idx  
        self.target_col_idx = 0  # Close Price is always Col 0
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 1. Extract raw data slices
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        # 2. Prepare Decoder Input (Start Token + Future Placeholder)
        seq_y_input = seq_y.copy()
        
        # --- MASKING LOGIC (STRICT REAL-WORLD SIMULATION) ---
        f_start = self.label_len 
        
        # 1. Save the last known Close Price (Target) and tomorrow's (t+1) Sentiment
        last_close = seq_y_input[f_start - 1, self.target_col_idx]
        if self.sent_col_idx is not None:
            tomorrow_sentiment = seq_y[f_start, self.sent_col_idx]
            
        # 2. Mask EVERYTHING in the prediction horizon to 0 (technicals, covariates, future sentiment)
        seq_y_input[f_start:, :] = 0.0
        
        # 3. Apply Persistence (last known value) strictly to the Close Price
        seq_y_input[f_start:, self.target_col_idx] = last_close
        
        # 4. Safely unmask ONLY tomorrow's (t+1) sentiment score
        if self.sent_col_idx is not None:
            seq_y_input[f_start, self.sent_col_idx] = tomorrow_sentiment

        # 3. Prepare Time Features
        seq_x_mark = self.dates[s_begin:s_end]
        seq_y_mark = self.dates[r_begin:r_end]

        # 4. Extract Ground Truth Targets for Loss calculation
        target = self.data[s_end:s_end + self.pred_len, 0:1]
        
        # Extract Sentiment Target (used for weighting loss)
        if self.sent_col_idx is not None:
            target_sentiment = self.data[s_end:s_end + self.pred_len, self.sent_col_idx:self.sent_col_idx+1]
        else:
            target_sentiment = np.zeros((self.pred_len, 1), dtype=np.float32)
            
        # Extract ADX Target (used for weighting loss)
        if self.adx_col_idx is not None:
             target_adx = self.data[s_end:s_end + self.pred_len, self.adx_col_idx:self.adx_col_idx+1]
        else:
             target_adx = np.zeros((self.pred_len, 1), dtype=np.float32)

        # 🚀 NEW: Extract the SINGLE regime label for the final prediction step 
        # (This represents the state of the market right before forecasting)
        target_regime = self.regimes[s_end - 1]

        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_x_mark, dtype=torch.float32),
                torch.tensor(seq_y_input, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(seq_y_mark, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(target_sentiment, dtype=torch.float32),
                torch.tensor(target_adx, dtype=torch.float32),
                torch.tensor(target_regime, dtype=torch.long)) # 🚀 Append Regime Label

class MultiFactorLoss(nn.Module):
    def __init__(self, neg_w=6.0, pos_w=2.0, neu_w=0.5, 
                 vol_high_w=2.0, vol_low_w=1.0, 
                 sent_thresh=0.05, adx_thresh=0.3, 
                 direction_w=1.0, reduction='mean'): # ⚙️ Added reduction parameter
        super(MultiFactorLoss, self).__init__()
        self.neg_w = neg_w
        self.pos_w = pos_w
        self.neu_w = neu_w
        self.vol_high_w = vol_high_w
        self.vol_low_w = vol_low_w
        self.sent_thresh = sent_thresh
        self.adx_thresh = adx_thresh
        self.direction_w = direction_w
        self.reduction = reduction 

    def forward(self, pred, target, sentiment, adx, last_known=None):
        # 1. Standard Weighted MSE Calculation
        squared_errors = (pred - target) ** 2
        
        sent_weights = torch.full_like(sentiment, self.neu_w)
        sent_weights[sentiment < -self.sent_thresh] = self.neg_w
        sent_weights[sentiment > self.sent_thresh] = self.pos_w
        
        vol_weights = torch.full_like(adx, self.vol_low_w)
        vol_weights[adx > self.adx_thresh] = self.vol_high_w
        
        final_weights = sent_weights * vol_weights
        final_weights = final_weights / (final_weights.mean() + 1e-8)
        
        mse_loss_raw = squared_errors * final_weights

        # 2. Directional Accuracy Penalty (Sample-wise)
        # 🔴 FIX (Issue A): Bypass directional penalty for Horizon=1 to prevent std() explosion
        if pred.size(1) == 1:
            combined_loss = mse_loss_raw
        else:
            if last_known is not None:
                # 🚀 ALIGNMENT FIX: Use anchor to penalize the most critical T=0 to T=1 transition
                first_diff_pred = pred[:, 0:1] - last_known
                first_diff_target = target[:, 0:1] - last_known
                diff_pred = torch.cat([first_diff_pred, pred[:, 1:] - pred[:, :-1]], dim=1)
                diff_target = torch.cat([first_diff_target, target[:, 1:] - target[:, :-1]], dim=1)
            else:
                # Fallback if no anchor is provided
                diff_pred = torch.cat([torch.zeros_like(pred[:, 0:1]), pred[:, 1:] - pred[:, :-1]], dim=1)
                diff_target = torch.cat([torch.zeros_like(target[:, 0:1]), target[:, 1:] - target[:, :-1]], dim=1)

            raw_direction_loss = torch.relu(-1.0 * (diff_pred * diff_target))
            # Scale penalty by target volatility to prevent gradient dominance
            std_per_sample = torch.std(diff_target, dim=1, keepdim=True)
            loss_direction_raw = raw_direction_loss / (std_per_sample + 1e-8)
            
            combined_loss = mse_loss_raw + (self.direction_w * loss_direction_raw)

        # 🚀 DRO / Standard Control: Returns raw tensor if 'none', else Scalar Mean
        if self.reduction == 'none':
            return combined_loss
        return torch.mean(combined_loss)
def load_and_split_data(target_col='Close', max_lookback=96):
    """
    Loads data, performs a static 60/20/20 Train/Val/Test split with proper overlaps,
    performs independent imputation, and detects market regimes using GMM.
    """
    ColorLog.extreme("LOADING RAW DATA FROM DISK", color="#00FFFF", border_color="#00FFFF")
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}. Generating mock data for testing...")
        dates = pd.date_range(start='1/1/2016', periods=2200, freq='D')
        
        mock_news_len = 3000 
        mock_news_dates = np.random.choice(dates, mock_news_len)
        mock_sentiments = np.random.uniform(-1, 1, mock_news_len)
        
        df_news = pd.DataFrame({
            'Date': mock_news_dates,
            'Sentiment_Score': mock_sentiments
        })
        
        df_stock = pd.DataFrame({
            'Date': dates,
            'Close': np.random.randn(2200).cumsum() + 100,
            'RSI': np.random.uniform(0, 100, 2200),
            'SMA': np.random.randn(2200).cumsum(),
            'ADX': np.random.uniform(0, 50, 2200),
            'Gold': np.random.randn(2200).cumsum(),
            'USDJPY': np.random.randn(2200).cumsum()
        })
        df_full = df_stock.merge(df_news.groupby('Date')['Sentiment_Score'].mean().reset_index(), on='Date', how='left')
        df_full['Sentiment_Score'] = df_full['Sentiment_Score'].fillna(0)
    else:
        df_raw = pd.read_csv(DATA_PATH)
        
        if 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        else:
            df_raw['Date'] = pd.to_datetime(df_raw.index)

        if 'Sentiment_Score' in df_raw.columns:
            df_grouped = df_raw.groupby('Date').agg(
                Sentiment_Score=('Sentiment_Score', 'mean')
            ).reset_index()
            
            df_technicals = df_raw.drop(columns=['Sentiment_Score']).groupby('Date').first().reset_index()
            df_full = pd.merge(df_technicals, df_grouped, on='Date', how='left')
            
            df_full['Sentiment_Score'] = df_full['Sentiment_Score'].fillna(0)
            if 'News_Volume' in df_full.columns:
                df_full['News_Volume'] = df_full['News_Volume'].fillna(0)
        else:
            df_full = df_raw.groupby('Date').first().reset_index()

    # 1. Guarantee chronological order before shifting
    df_full = df_full.sort_values('Date').reset_index(drop=True)
        
    # 2. Shift the sentiment so today's news predicts tomorrow's price
    if 'Sentiment_Score' in df_full.columns:
        df_full['Sentiment_Score'] = df_full['Sentiment_Score'].shift(1).fillna(0)
    # ⚙️ STATIC SPLIT LOGIC (60/20/20)
    n = len(df_full)
    val_start_idx = int(n * 0.60)
    test_start_idx = int(n * 0.70) 
    
    train_slice = slice(0, val_start_idx)
    val_slice = slice(max(0, val_start_idx - max_lookback), test_start_idx)
    test_slice = slice(max(0, test_start_idx - max_lookback), None)
    
    val_overlap_len = val_start_idx - val_slice.start
    test_overlap_len = test_start_idx - test_slice.start

    ColorLog.extreme("DETECTING MARKET REGIMES (GMM)", color="#8A2BE2", border_color="#8A2BE2")
    # ⚙️ REGIME DETECTION (GMM based on Rolling Volatility)
    # Calculate log returns and 20-day rolling volatility
    returns = np.diff(np.log(df_full[target_col].values + 1e-8))
    volatility_raw = pd.Series(returns).rolling(window=20).std().bfill().values.reshape(-1, 1)
    
    # Padding: np.diff reduces length by 1, so we pad the first value to match df_full
    volatility = np.vstack([volatility_raw[0], volatility_raw]) 

    # --- LEAKAGE FIX: Fit ONLY on Training Data ---
    train_end = train_slice.stop

    # Fit GMM to identify 3 clusters using ONLY historical training volatility
    gmm = GaussianMixture(n_components=3, random_state=42).fit(volatility[:train_end])
    
    # Predict regimes for the entire dataset using the model fitted on 'past' data
    df_full['Regime'] = gmm.predict(volatility)
    # -----------------------------------------------

    current_regime_idx = df_full['Regime'].iloc[test_slice.start]
    regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
    print(f"Detected Market Regime for this window: {regime_names.get(current_regime_idx, 'Unknown')}")

    # Ensure Target is at Index 0
    cols = list(df_full.columns)
    if target_col in cols:
        cols.insert(0, cols.pop(cols.index(target_col)))
    df_full = df_full[cols]
    
    # Identify Indices (Excluding 'Regime' from features to prevent leakage)
    feature_cols = [c for c in df_full.columns if c not in ['Date', 'Regime']]
    
    def get_numpy_idx(col_name):
        if col_name in feature_cols:
            return feature_cols.index(col_name)
        return None

    sent_col_idx = get_numpy_idx('Sentiment_Score')
    adx_col_idx = get_numpy_idx('ADX')
    
    time_marks = time_features(df_full['Date'].dt)
    
    # 🚀 NEW: Extract GMM Regime Labels as a continuous array before splitting
    regime_labels_full = df_full['Regime'].values

    ColorLog.extreme("CLEANING DATA & IMPUTING MISSING VALUES", color="#FF8C00", border_color="#FF8C00")
    # =========================================================
    # ⚙️ MINIMAL LEAKAGE FIX: Impute safely BEFORE splitting
    # =========================================================
    numeric_cols = df_full.select_dtypes(include=[np.number]).columns
    
    # 1. Forward-fill globally (100% safe, past data flows into future)
    df_full[numeric_cols] = df_full[numeric_cols].ffill()
    
    # 2. Backward-fill ONLY the training slice (Prevents future-leakage into test set)
    df_full.iloc[:train_slice.stop, df_full.columns.get_indexer(numeric_cols)] = \
        df_full.iloc[:train_slice.stop][numeric_cols].bfill()
        
    # 3. Catch-all for entirely empty columns
    df_full[numeric_cols] = df_full[numeric_cols].fillna(0)
    # =========================================================

    # Split DataFrames (Now safely imputed)
    df_train = df_full.iloc[train_slice].copy()
    df_val = df_full.iloc[val_slice].copy()
    df_test = df_full.iloc[test_slice].copy()

    ColorLog.extreme("SCALING FEATURES & TARGETS", color="#7FFF00", border_color="#7FFF00")
    # Scaling
    train_raw = df_train[feature_cols].values
    val_raw = df_val[feature_cols].values
    test_raw = df_test[feature_cols].values

    # Covariate Scaling (Excluding target at index 0)
    scaler_cov = MinMaxScaler()
    train_cov_scaled = scaler_cov.fit_transform(train_raw[:, 1:])
    val_cov_scaled = scaler_cov.transform(val_raw[:, 1:])
    test_cov_scaled = scaler_cov.transform(test_raw[:, 1:])

    # Revert Sentiment Scaling if applicable (preserving raw LLM scores for weighting)
    if sent_col_idx is not None and sent_col_idx > 0:
        cov_sent_idx = sent_col_idx - 1 
        for cov_chunk, raw_chunk in zip([train_cov_scaled, val_cov_scaled, test_cov_scaled], 
                                        [train_raw, val_raw, test_raw]):
            cov_chunk[:, cov_sent_idx] = raw_chunk[:, sent_col_idx]

    # Target Scaling (Index 0)
    scaler_target = MinMaxScaler()
    train_target_scaled = scaler_target.fit_transform(train_raw[:, 0:1])
    val_target_scaled = scaler_target.transform(val_raw[:, 0:1])
    test_target_scaled = scaler_target.transform(test_raw[:, 0:1])

    # Recombine target and covariates
    train_vals = np.hstack([train_target_scaled, train_cov_scaled])
    val_vals = np.hstack([val_target_scaled, val_cov_scaled])
    test_vals = np.hstack([test_target_scaled, test_cov_scaled])
    
    print(f"Final Shapes: Train {train_vals.shape} | Val {val_vals.shape} | Test {test_vals.shape}")

    # 🚀 NEW: Pass the sliced GMM Regime labels out
    return (train_vals, time_marks[train_slice], regime_labels_full[train_slice],
            val_vals, time_marks[val_slice], regime_labels_full[val_slice],
            test_vals, time_marks[test_slice], regime_labels_full[test_slice],
            len(feature_cols), scaler_target, scaler_cov, 
            sent_col_idx, adx_col_idx, val_overlap_len, test_overlap_len)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed_month = nn.Embedding(13, d_model)
        self.embed_day = nn.Embedding(32, d_model)
        self.embed_weekday = nn.Embedding(7, d_model)
        self.embed_hour = nn.Embedding(25, d_model)
        
    def forward(self, x):
        # Safely handle potential NaNs before casting to integer to prevent silent dimension errors
        x = torch.nan_to_num(x, nan=0.0).long()
        return (self.embed_month(x[:, :, 0]) + self.embed_day(x[:, :, 1]) +
                self.embed_weekday(x[:, :, 2]) + self.embed_hour(x[:, :, 3]))

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, use_temporal=True, use_positional=True):
        super(DataEmbedding, self).__init__()
        self.use_temporal = use_temporal
        self.use_positional = use_positional # ⚙️ NEW: Ablation flag
        
        self.value_embedding = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                         kernel_size=3, padding=1, padding_mode='circular')
        
        # ⚙️ Modular initialization
        if self.use_positional:
            self.position_embedding = PositionalEmbedding(d_model)
        if self.use_temporal:
            self.temporal_embedding = TimeFeatureEmbedding(d_model)
            
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 1. Base Value Embedding
        x = self.value_embedding(x.permute(0, 2, 1)).transpose(1, 2)
        
        # 2. Conditional Positional Bias
        if self.use_positional:
            x = x + self.position_embedding(x)
            
        # 3. Conditional Temporal Bias
        if self.use_temporal:
            x = x + self.temporal_embedding(x_mark)
            
        return self.dropout(x)

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_k, x_v, attn_mask=None):
        B, L_Q, _ = x_q.shape
        _, L_K, _ = x_k.shape
        Q = self.q_proj(x_q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x_k).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x_v).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        # Sampling factor c*ln(L)
        u = int(self.factor * math.log(L_Q)) if L_Q > 0 else 1
        u = min(max(u, 1), L_Q)
        sample_k = min(u, L_K)
        rand_idx = torch.randint(0, L_K, (B, self.n_heads, sample_k), device=Q.device)
        idx_expanded = rand_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        K_sample = torch.gather(K, 2, idx_expanded)

        # Q_K_sample
        scores_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) * self.scale

        # Mask handling for ProbSparse
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask_full = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, L_Q, L_K)
            else:
                mask_full = attn_mask
            
            mask_gather_idx = rand_idx.unsqueeze(2).expand(-1, -1, L_Q, -1)
            mask_sample = torch.gather(mask_full, -1, mask_gather_idx)
            scores_sample = scores_sample.masked_fill(mask_sample, -1e4)

            M_max = torch.max(scores_sample, dim=-1)[0]
            
            valid_counts = (mask_sample == False).sum(dim=-1).float().clamp(min=1.0)
            scores_clean = scores_sample.clone()
            scores_clean[mask_sample] = 0.0
            M_mean = scores_clean.sum(dim=-1) / valid_counts
            M_score = M_max - M_mean
        else:
            M_max = torch.max(scores_sample, dim=-1)[0]
            M_mean = torch.mean(scores_sample, dim=-1)
            M_score = M_max - M_mean

        _, top_u_idx = torch.topk(M_score, u, dim=-1)
        top_u_idx_expanded = top_u_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        
        # Select Top-u Queries
        Q_selected = torch.gather(Q, 2, top_u_idx_expanded)

        # Computed Selected Attention
        scores_selected = torch.matmul(Q_selected, K.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            gather_idx_mask = top_u_idx.unsqueeze(-1).expand(-1, -1, -1, L_K)
            mask_selected = torch.gather(mask_full, 2, gather_idx_mask)
            scores_selected = scores_selected.masked_fill(mask_selected, -1e4)

        attn_weights = torch.softmax(scores_selected, dim=-1)
        context_selected = torch.matmul(attn_weights, V)

        # --- CRITICAL FIX: Causal / Cumulative Mean Assignment for unselected queries ---
        if attn_mask is not None:
            # 🔴 FIX (Issue B): Mask-aware cumulative mean to prevent bias from padded/masked tokens
            # We assume mask_full has True for invalid positions.
            valid_mask = (~mask_full[:, :, 0, :]).unsqueeze(-1) if mask_full.dim() == 4 else (~mask_full).unsqueeze(-1)
            
            V_valid = V.masked_fill(~valid_mask, 0.0)
            cum_v = V_valid.cumsum(dim=2)
            valid_counts = valid_mask.float().cumsum(dim=2).clamp(min=1.0)
            
            # V_mean is now temporally accurate AND mask-correct per step
            context = (cum_v / valid_counts).clone()
        else:
            # Safe to do global mean in the Encoder
            V_mean = V.mean(dim=2, keepdim=True) 
            context = V_mean.expand(B, self.n_heads, L_Q, self.d_k).clone()

        # Scatter the selected queries into the context
        context.scatter_(2, top_u_idx_expanded, context_selected) 

        context = context.transpose(1, 2).contiguous().view(B, L_Q, -1)
        return self.out_proj(context)

class DistillingLayer(nn.Module):
    def __init__(self, d_model):
        super(DistillingLayer, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model) # Changed to LayerNorm
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x starts as [Batch, SeqLen, d_model]
        x = x.permute(0, 2, 1)       # [Batch, d_model, SeqLen] for Conv1d
        x = self.conv(x)
        x = x.permute(0, 2, 1)       # [Batch, SeqLen, d_model] for LayerNorm
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)       # [Batch, d_model, SeqLen] for MaxPool1d
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)       # Back to [Batch, SeqLen, d_model]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, prob_sparse=True, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.prob_sparse = prob_sparse
        if prob_sparse:
            self.attn = ProbSparseAttention(d_model, n_heads)
        else:
            # Full Attention (Canonical)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.prob_sparse:
            new_x = self.attn(x, x, x)
        else:
            new_x, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(new_x))
        new_x = self.ff(x)
        x = self.norm2(x + self.dropout(new_x))
        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, e_layers=[3, 2, 1], distil=True, prob_sparse=True, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.stacks = nn.ModuleList()
        self.distil = distil
        for num_layers in e_layers:
            layers = []
            for i in range(num_layers):
                layers.append(EncoderLayer(d_model, n_heads, prob_sparse, dropout=dropout))
                # Distilling is applied if distil=True AND it's not the last layer of the stack
                if self.distil and i < num_layers - 1:
                    layers.append(DistillingLayer(d_model))
            self.stacks.append(nn.Sequential(*layers))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        stack_outs = []
        for i, stack in enumerate(self.stacks):
            # Pyramid input: downsample input length for each stack
            inp_len = x.shape[1] // (2**i)
            inp = x[:, -inp_len:, :]
            out = stack(inp)
            stack_outs.append(out)
        
        # If distilling is used, stacks output different lengths. Truncate to min.
        min_len = min([s.shape[1] for s in stack_outs])
        aligned_outs = [s[:, -min_len:, :] for s in stack_outs]
        out = torch.cat(aligned_outs, dim=1)
        out = self.norm(out)
        return out

class DecoderLayer(nn.Module):
    """
    Updated to strictly follow the Informer paper: 
    ProbSparse is ONLY used for Self-Attention. 
    Cross-Attention always uses full Multi-Head Attention.
    """
    def __init__(self, d_model, n_heads, prob_sparse=True, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.prob_sparse = prob_sparse
        
        # 1. Self-Attention (Can be ProbSparse or Full)
        if prob_sparse:
            self.self_attn = ProbSparseAttention(d_model, n_heads)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            
        # 2. Cross-Attention (ALWAYS Full Multi-head Attention per original paper)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, tgt_mask=None, cross_attn=True):
        # --- Self Attention Block ---
        if self.prob_sparse:
            new_x = self.self_attn(x, x, x, attn_mask=tgt_mask)
        else:
            new_x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(new_x))
        
        # --- Cross Attention Block using Encoder Outputs ---
        if cross_attn:
            # ALWAYS use standard Multi-head attention here
            new_x, _ = self.cross_attn(x, cross, cross)
            x = self.norm2(x + self.dropout(new_x))
        
        # --- Feed Forward Block ---
        new_x = self.ff(x)
        if cross_attn:
            x = self.norm3(x + self.dropout(new_x))
        else:
            x = self.norm2(x + self.dropout(new_x))
            
        return x
class PatchTSTDataset(Dataset):
    def __init__(self, data, regimes, seq_len, pred_len, sent_col_idx=None, adx_col_idx=None):
        self.data = data
        self.regimes = regimes # 🚀 NEW
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sent_col_idx = sent_col_idx
        self.adx_col_idx = adx_col_idx
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin, s_end = index, index + self.seq_len
        seq_x = self.data[s_begin:s_end]
        target = self.data[s_end:s_end + self.pred_len, 0:1]
        
        sent = self.data[s_end:s_end + self.pred_len, self.sent_col_idx:self.sent_col_idx+1] if self.sent_col_idx is not None else np.zeros((self.pred_len, 1))
        adx = self.data[s_end:s_end + self.pred_len, self.adx_col_idx:self.adx_col_idx+1] if self.adx_col_idx is not None else np.zeros((self.pred_len, 1))

        target_regime = self.regimes[s_end - 1] # 🚀 NEW
        
        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(sent, dtype=torch.float32),
                torch.tensor(adx, dtype=torch.float32),
                torch.tensor(target_regime, dtype=torch.long)) # 🚀 NEW

class TransformerEncoderLayerLayerNorm(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Changed to LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        # src shape: [Batch*Channels, Num_Patches, d_model]
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        
        # LayerNorm expects [..., d_model], so no permutation is needed!
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        src = self.norm2(src)
        
        return src
class PatchTST(nn.Module):
    # ⚙️ MINIMAL CHANGE: Added enable_revin=True to the parameters
    def __init__(self, c_in, seq_len, pred_len, patch_len=16, stride=8, d_model=64, n_heads=4, e_layers=3, dropout=0.1, enable_revin=True):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride 
        self.pred_len = pred_len
        
        self.patch_num = (seq_len - self.patch_len) // stride + 1
        
        # ⚙️ MINIMAL CHANGE: Passed the toggle to RevIN
        self.revin = RevIN(c_in, affine=True, enable_revin=enable_revin)
        
        # Channel Independence: Shared embeddings across all channels
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.patch_num + 2, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # ⚙️ Updated to call the new LayerNorm class name
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayerLayerNorm(d_model, n_heads, d_model * 4, dropout) 
            for _ in range(e_layers)
        ])
        
        # Final Prediction Head
        self.head = nn.Linear(self.patch_num * d_model, pred_len)
        self.pretrain_head = nn.Linear(d_model, patch_len)
        
    def forward(self, x_enc, pretrain=False, mask_ratio=0.4):
        B, L, C = x_enc.shape
        x_enc, mean, stdev, gamma, beta = self.revin(x_enc, 'norm')

        # Generate Patches
        x_enc_p = x_enc.permute(0, 2, 1) 
        patches = x_enc_p.unfold(dimension=2, size=self.patch_len, step=self.stride) 
        
        curr_patch_num = patches.shape[2] 
        patches_orig = patches.clone()
        
        if pretrain:
            mask = (torch.rand(B, C, curr_patch_num, device=x_enc.device) < mask_ratio)
            patches = patches.masked_fill(mask.unsqueeze(-1), 0.0)

        # Fold Channel dimension into Batch dimension: [B*C, curr_patch_num, patch_len]
        patches = patches.reshape(B * C, curr_patch_num, self.patch_len)
        pos_emb = self.position_embedding[:, :curr_patch_num, :]
        
        enc_in = self.dropout(self.patch_embedding(patches) + pos_emb)
        enc_out = self.encoder(enc_in) 
        
        if pretrain:
            rec_patches = self.pretrain_head(enc_out).reshape(B, C, curr_patch_num, self.patch_len)
            return rec_patches, patches_orig, mask 

        # Interpolate if sequence length caused a patch count mismatch
        if curr_patch_num != self.patch_num:
            enc_out = enc_out.permute(0, 2, 1)
            enc_out = F.interpolate(enc_out, size=self.patch_num, mode='linear', align_corners=False)
            enc_out = enc_out.permute(0, 2, 1)

        enc_out = enc_out.reshape(B, C, self.patch_num, self.d_model)
        
        # Extract only the target variable (Close price, index 0)
        target_enc_out = enc_out[:, 0, :, :]
        
        target_enc_out = target_enc_out.reshape(B, self.patch_num * self.d_model)
        dec_out = self.head(target_enc_out).unsqueeze(-1) 
        
        # Denormalize explicitly using only the 0th index stats from RevIN
        return self.revin(dec_out, 'denorm', mean=mean, stdev=stdev, gamma=gamma, beta=beta, target_idx=0)
class Informer(nn.Module):
    # ⚙️ MINIMAL CHANGE: Added seq_len and norm_type to parameters
    def __init__(self, input_dim, seq_len=96, d_model=64, n_heads=4, out_len=1, 
                 e_layers=[3, 2, 1], d_layers=2, distil=True, prob_sparse=True, 
                 dec_only=False, gen_dec=True, dropout=0.1, 
                 use_temporal=True, use_positional=True, 
                 enable_revin=True, norm_type='revin'):
        super(Informer, self).__init__()
        self.pred_len, self.dec_only, self.gen_dec = out_len, dec_only, gen_dec

        # ⚙️ MINIMAL CHANGE: Route the correct normalization module to 'self.revin'
        if norm_type == 'san':
            self.revin = SANorm(input_dim, seq_len=seq_len)
        elif norm_type == 'fan':
            self.revin = FANorm(input_dim)
        elif norm_type == 'wdan':
            self.revin = WDANorm(input_dim)
        elif norm_type == 'cora':
            self.revin = CoRANorm(input_dim)
        else:
            # Fallback to standard RevIN
            self.revin = RevIN(input_dim, affine=True, enable_revin=enable_revin)

        # Embeddings (⚙️ Now passing both temporal and positional toggles)
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout, use_temporal, use_positional)
        self.dec_embedding = DataEmbedding(input_dim, d_model, dropout, use_temporal, use_positional)

        # Encoder stack
        if not self.dec_only:
            self.encoder = InformerEncoder(d_model, n_heads, e_layers=e_layers, distil=distil, 
                                           prob_sparse=prob_sparse, dropout=dropout)

        # Decoder stack
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, prob_sparse=prob_sparse, dropout=dropout) 
            for _ in range(d_layers)
        ])
        
        self.dec_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def _get_norm_stats(self, x_enc, x_dec):
        """Standardizes RevIN normalization logic."""
        if self.dec_only:
            x_dec, mean, std, gamma, beta = self.revin(x_dec, 'norm')
            return None, x_dec, mean, std, gamma, beta
        
        x_enc, mean, std, gamma, beta = self.revin(x_enc, 'norm')
        x_dec, _, _, _, _ = self.revin(x_dec, 'norm', mean=mean, stdev=std, gamma=gamma, beta=beta)
        return x_enc, x_dec, mean, std, gamma, beta

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, training_target=None):
        x_enc, x_dec, mean, std, gamma, beta = self._get_norm_stats(x_enc, x_dec)

        enc_out = None
        if not self.dec_only:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out = self.encoder(enc_out)

        if self.gen_dec:
            dec_out = self._generative_decode(x_dec, x_mark_dec, enc_out)
        else:
            dec_out = self._dynamic_decode(x_dec, x_mark_dec, enc_out, training_target, mean, std, gamma, beta)

        return self.revin(dec_out, 'denorm', mean=mean, stdev=std, gamma=gamma, beta=beta, target_idx=0)

    def _generative_decode(self, x_dec, x_mark_dec, enc_out):
        x = self.dec_embedding(x_dec, x_mark_dec)
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask=mask, cross_attn=not self.dec_only)
        return self.projection(self.dec_norm(x)[:, -self.pred_len:, :])

    def _dynamic_decode(self, x_dec, x_mark_dec, enc_out, training_target, mean, std, gamma=None, beta=None):
        if self.training and training_target is not None:
            # 🔴 FIX (Issue C) Architectural Note: Normalizing decoder targets using encoder stats.
            # Assumption: The target sequence distribution strongly matches the lookback sequence.
            # This holds for continuous time-series, but under abrupt regime shifts, local normalization may drift.
            x_tf, _, _, _, _ = self.revin(training_target[:, :-1, :], 'norm', mean=mean, stdev=std, gamma=gamma, beta=beta)
            x = self.dec_embedding(x_tf, x_mark_dec[:, :-1, :])
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
            for layer in self.decoder:
                x = layer(x, enc_out, tgt_mask=mask, cross_attn=not self.dec_only)
            return self.projection(self.dec_norm(x)[:, -self.pred_len:, :])
        
        label_len = x_dec.shape[1] - self.pred_len
        buf, preds = x_dec.clone(), torch.zeros(x_dec.size(0), self.pred_len, 1, device=x_dec.device)
        for i in range(self.pred_len):
            curr_len = label_len + i
            x = self.dec_embedding(buf[:, :curr_len, :], x_mark_dec[:, :curr_len, :])
            mask = torch.triu(torch.ones(curr_len, curr_len), diagonal=1).bool().to(x.device)
            for layer in self.decoder:
                x = layer(x, enc_out, tgt_mask=mask, cross_attn=not self.dec_only)
            step_pred = self.projection(self.dec_norm(x)[:, -1:, :]) 
            preds[:, i:i+1, :] = step_pred
            if i < self.pred_len - 1:
                buf[:, curr_len:curr_len+1, 0:1] = step_pred
        return preds

# ==========================================
# 4. TRAINING WRAPPERS
# ==========================================

class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                ColorLog.extreme(f"EARLY STOPPING TRIGGERED! (Patience: {self.patience} reached)", color="#FF0000", border_color="#FF0000")
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_torch_model(model, train_loader, val_loader, epochs, lr, patience, model_save_path,
                      scaler_target, 
                      weight_decay=1e-5, 
                      use_weighted_loss=False, enable_custom_loss=True, 
                      neg_w=2.0, pos_w=1.5, neu_w=0.5, 
                      vol_high_w=2.0, vol_low_w=1.0, 
                      direction_w=1.0, trial=None, use_dro=False, adx_idx=None, scaler_cov=None): # 🚀 Added scaler_cov 
    """
    Robust training loop that computes pure training performance using calculate_metrics
    with support for weight_decay regularization and Distributionally Robust Optimization (DRO).
    """
    # ⚙️ Pass weight_decay to the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 🚀 FIX: Convert economic ADX thresholds (20, 25) into the scaled domain ONCE
    thresh_20, thresh_25 = 0.15, 0.30 # Default fallbacks
    if adx_idx is not None and scaler_cov is not None:
        try:
            cov_idx = adx_idx - 1 # Adjust index for covariate scaler
            # Ensure the scaler was properly fitted before trying to access min/max arrays
            if hasattr(scaler_cov, 'data_min_') and hasattr(scaler_cov, 'data_max_'):
                adx_min = scaler_cov.data_min_[cov_idx]
                adx_max = scaler_cov.data_max_[cov_idx]
                thresh_20 = (20.0 - adx_min) / (adx_max - adx_min + 1e-8)
                thresh_25 = (25.0 - adx_min) / (adx_max - adx_min + 1e-8)
        except Exception as e:
            ColorLog.warn(f"Failed to calculate dynamic ADX thresholds, using defaults. Error: {e}")

    # 🚀 FIX: Pass 'none' reduction to criterion if DRO is active so we can group losses per regime
    reduction_mode = 'none' if use_dro else 'mean'

    if use_weighted_loss and enable_custom_loss:
        criterion = MultiFactorLoss(neg_w=neg_w, pos_w=pos_w, neu_w=neu_w,
                                    vol_high_w=vol_high_w, vol_low_w=vol_low_w, 
                                    adx_thresh=thresh_25, # 🚀 Link custom loss to economic threshold
                                    direction_w=direction_w, reduction=reduction_mode)
    else:
        criterion = nn.MSELoss(reduction=reduction_mode)

    early_stopper = EarlyStopping(patience=patience) if val_loader is not None else None
    best_loss = float('inf')
    best_epoch = 0  

    ColorLog.extreme(f"STARTING NEURAL NETWORK TRAINING LOOP ({epochs} Epochs max)", color="#FF6347", border_color="#FF6347")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_acts = [], []
        epoch_grad_norms = [] 

        for batch in train_loader:
            optimizer.zero_grad()
            batch = [b.to(DEVICE) for b in batch]
            
            # 🚀 NEW: Unpack the batch dynamically based on the true GMM regime dataset lengths
            if len(batch) == 9:
                enc, enc_mark, dec_input, dec_true, dec_mark, y, sent_target, adx_target, batch_regimes = batch
                pred = model(enc, enc_mark, dec_input, dec_mark, training_target=dec_true)
            else:
                enc, y, sent_target, adx_target, batch_regimes = batch
                pred = model(enc)

            if use_dro:
                # 🚀 DRO LOGIC: Calculate loss without reducing, then group by regime
                if use_weighted_loss and enable_custom_loss:
                    last_known_scaled = enc[:, -1:, 0:1]
                    raw_losses = criterion(pred, y, sent_target, adx_target, last_known=last_known_scaled)
                else:
                    raw_losses = criterion(pred, y)
                
                # ⚙️ CRITICAL FIX: Group raw losses directly by the GMM Cluster Labels
                regime_losses = []
                for i in range(3):
                    mask = (batch_regimes == i)
                    if mask.any():
                        regime_losses.append(raw_losses[mask].mean())
                
                # 🧠 Distributionally Robust Optimization: Minimize the worst-case true market regime
                loss = torch.stack(regime_losses).max() if regime_losses else raw_losses.mean()
            else:
                # STANDARD LOGIC
                if use_weighted_loss and enable_custom_loss:
                    last_known_scaled = enc[:, -1:, 0:1]
                    loss = criterion(pred, y, sent_target, adx_target, last_known=last_known_scaled)
                else:
                    loss = criterion(pred, y)

            loss.backward()
            
            if torch.isnan(loss) or torch.isinf(loss):
                ColorLog.warn("Exploding gradients (NaN loss) detected. Aborting this bad trial early.")
                return float('inf')
            
            # 5️⃣ Gradient Monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_grad_norms.append(grad_norm.item())
            
            optimizer.step()
            train_loss += loss.item()
            
            train_preds.append(pred.detach().cpu().numpy())
            train_acts.append(y.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = np.mean(epoch_grad_norms)
        
        train_preds_concat = np.concatenate(train_preds)
        train_acts_concat = np.concatenate(train_acts)
        _, _, train_pure_scaled, _ = calculate_metrics(train_preds_concat, train_acts_concat, scaler_target)

        val_mse_scaled = 0.0
        if val_loader is not None:
            model.eval()
            val_preds, val_acts = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = [b.to(DEVICE) for b in batch]
                    # 🚀 NEW: Match validation unpacking lengths
                    if len(batch) == 9:
                        enc, enc_mark, dec_input, dec_true, dec_mark, y, sent_target, adx_target, batch_regimes = batch
                        # 🚀 EVALUATION FIX: Disable Teacher Forcing during validation (training_target=None)
                        pred = model(enc, enc_mark, dec_input, dec_mark, training_target=None)
                    else:
                        enc, y, sent_target, adx_target, batch_regimes = batch
                        pred = model(enc)
                    
                    val_preds.append(pred.cpu().numpy())
                    val_acts.append(y.cpu().numpy())

            val_preds = np.concatenate(val_preds)
            val_acts = np.concatenate(val_acts)
            _, _, val_mse_scaled, _ = calculate_metrics(val_preds, val_acts, scaler_target)

            scheduler.step(val_mse_scaled)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    ↳ Epoch [{epoch+1}/{epochs}] | Train Grad Loss: {avg_train_loss:.6f} | Train Scaled MSE: {train_pure_scaled:.6f}% | Val Scaled MSE: {val_mse_scaled:.6f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_mse_scaled < best_loss:
                best_loss = val_mse_scaled
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)

            if trial is not None:
                trial.report(val_mse_scaled, epoch)
                trial.set_user_attr("best_epoch", best_epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            early_stopper(val_mse_scaled)
            
            if mlflow.active_run() and trial is None:
                mlflow.log_metrics({
                    "Train_Scaled_MSE": train_pure_scaled,
                    "Val_Scaled_MSE": val_mse_scaled,
                    "Train_Grad_Loss": avg_train_loss,
                    "Epoch_Grad_Norm": avg_grad_norm
                }, step=epoch)

            if early_stopper.early_stop:
                break
        else:
            torch.save(model.state_dict(), model_save_path)
            best_loss = avg_train_loss

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        
    return best_loss

def calculate_metrics(preds_scaled, actuals_scaled, scaler_target):
    if np.isnan(preds_scaled).any() or np.isinf(preds_scaled).any():
        ColorLog.warn("NaNs or Infs detected in predictions! Returning infinity to prune trial.")
        return float('inf'), float('inf'), float('inf'), float('inf')

    orig_shape = preds_scaled.shape
    flat_preds = preds_scaled.reshape(-1, 1)
    flat_actuals = actuals_scaled.reshape(-1, 1)
    
    # 1. Inverse transform back to original prices (e.g., Dollars)
    real_pred = scaler_target.inverse_transform(flat_preds).reshape(orig_shape)
    real_act = scaler_target.inverse_transform(flat_actuals).reshape(orig_shape)
    
    # 2. Calculate errors in Real Dollars
    mse_real = mean_squared_error(real_act.flatten(), real_pred.flatten())
    mae_real = mean_absolute_error(real_act.flatten(), real_pred.flatten())
    
    # 3. Calculate errors in Scaled units (multiplied by 100 as per IEEE standard)
    mse_scaled = mean_squared_error(actuals_scaled.flatten(), preds_scaled.flatten()) * 100
    mae_scaled = mean_absolute_error(actuals_scaled.flatten(), preds_scaled.flatten()) * 100
    
    return mse_real, mae_real, mse_scaled, mae_scaled

def calculate_naive_metrics(test_data, pred_len, scaler_target, fixed_start_idx):
    """
    Calculates Naive metrics aligned to a fixed start index (Max Lookback).
    
    Args:
        test_data: The scaled test dataset.
        pred_len: Forecast horizon.
        scaler_target: Scaler to inverse transform predictions.
        fixed_start_idx: The index where predictions MUST start (e.g., 96).
                         This ensures we skip the 'warm-up' period of the largest model.
    """
    target_data = test_data[:, 0]
    naive_preds = []
    naive_actuals = []

    # Iterate starting exactly from the fixed_start_idx (e.g., 96)
    # This aligns the first prediction to be for day 97 (index 96)
    # using day 96 (index 95) as the history.
    for i in range(fixed_start_idx, len(target_data) - pred_len+1):
        # 1. Last Known Value (Persistence)
        # We use the value immediately preceding the prediction window (i-1)
        last_known_val = target_data[i-1]
        
        # 2. Ground Truth Window
        ground_truth = target_data[i : i+pred_len]
        
        # 3. Create Prediction (Flat Line for the horizon)
        pred_seq = np.full_like(ground_truth, last_known_val)
        
        naive_preds.append(pred_seq)
        naive_actuals.append(ground_truth)
        
    if len(naive_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0 # ✅ MINIMAL CHANGE: Added 4th zero to match new return signature

    naive_preds = np.array(naive_preds)
    naive_actuals = np.array(naive_actuals)
    
    # ✅ MINIMAL CHANGE: Swapped to calculate_metrics
    return calculate_metrics(naive_preds, naive_actuals, scaler_target)
def setup_scenario_context(scenario, train_vals_full, train_regimes_full, 
                           val_vals_full, val_regimes_full, 
                           test_vals_full, test_regimes_full, 
                           input_dim_full, sent_idx_full, adx_idx_full, scaler_cov): 
    if scenario == "Without Sentiment":
        # Exclude sentiment column (assumed last)
        train_data = train_vals_full[:, :-1]
        val_data = val_vals_full[:, :-1]
        test_data = test_vals_full[:, :-1]
        input_dim = input_dim_full - 1
        use_weighted_loss = False
        current_sent_idx = None
        
        # Adjust ADX index if it was originally after the sentiment column
        if adx_idx_full is not None and sent_idx_full is not None:
            current_adx_idx = adx_idx_full - 1 if adx_idx_full > sent_idx_full else adx_idx_full
        else:
            current_adx_idx = adx_idx_full
    else:
        train_data = train_vals_full
        val_data = val_vals_full
        test_data = test_vals_full
        input_dim = input_dim_full
        use_weighted_loss = True
        current_sent_idx = sent_idx_full
        current_adx_idx = adx_idx_full

    # 🚀 NEW: Pass the untouched regimes alongside the data
    return (train_data, train_regimes_full, 
            val_data, val_regimes_full, 
            test_data, test_regimes_full, 
            input_dim, use_weighted_loss, current_sent_idx, current_adx_idx, scaler_cov)
def evaluate_naive_baseline(scenario, pred_len, test_data, scaler_target, test_overlap_len, OUTPUT_FILE):
    ColorLog.extreme(f"EVALUATING NAIVE BASELINE (H={pred_len})", color="#A9A9A9", border_color="#A9A9A9")
    # 🎨 Elegant Header
    # NEW
    cprint(f"<br>{ColorLog.OKCYAN}➤ Scenario: {scenario} | Horizon: {pred_len} | Method: Naive Persistence{ColorLog.ENDC}")    
    target_data = test_data[:, 0]
    naive_preds = []
    naive_actuals = []

    # Calculate window by window
    log_str = ""
    for i in range(test_overlap_len, len(target_data) - pred_len + 1):
        last_known_val = target_data[i-1]
        ground_truth = target_data[i : i+pred_len]
        pred_seq = np.full_like(ground_truth, last_known_val)
        
        real_pred = scaler_target.inverse_transform(pred_seq.reshape(-1, 1)).flatten()
        real_actual = scaler_target.inverse_transform(ground_truth.reshape(-1, 1)).flatten()
        
        window_idx = i - test_overlap_len + 1
        # Only log the first 3 and last 3 to keep the notebook memory safe and clean
        total_windows = len(target_data) - pred_len + 1 - test_overlap_len
        if window_idx <= 3 or window_idx >= total_windows - 2:
            log_str += f"&nbsp;&nbsp;{ColorLog.OKBLUE}Window {window_idx:03d}{ColorLog.ENDC} | Forecast: {real_pred} | Actual: {real_actual}<br>"
        elif window_idx == 4:
            log_str += f"&nbsp;&nbsp;{ColorLog.WARNING}... skipping {total_windows - 6} middle windows ...{ColorLog.ENDC}<br>"
            
        naive_preds.append(pred_seq)
        naive_actuals.append(ground_truth)
    
    # Print the aggregated string exactly ONCE
    cprint(log_str)
    
    naive_preds = np.array(naive_preds).reshape(-1, pred_len, 1)
    naive_actuals = np.array(naive_actuals).reshape(-1, pred_len, 1)
    
    # Calculate global metrics
    mse_real, mae_real, mse_scaled, mae_scaled = calculate_metrics(naive_preds, naive_actuals, scaler_target)
    
    # 🎨 Elegant Results Summary (⚙️ MINIMAL CHANGE: Updated to .6f formatting)
    ColorLog.success(f"Naive Baseline Baseline Results Summary:")
    cprint(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ColorLog.BOLD}Test Scaled MSE (%):{ColorLog.ENDC} {ColorLog.OKGREEN}{mse_scaled:.6f}%{ColorLog.ENDC}")
    cprint(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ColorLog.BOLD}Test Scaled MAE (%):{ColorLog.ENDC} {ColorLog.OKGREEN}{mae_scaled:.6f}%{ColorLog.ENDC}")
    cprint(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ColorLog.BOLD}Test Real MSE:{ColorLog.ENDC} {ColorLog.OKGREEN}{mse_real:.6f}{ColorLog.ENDC}")
    cprint(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ColorLog.BOLD}Test Real MAE:{ColorLog.ENDC} {ColorLog.OKGREEN}{mae_real:.6f}{ColorLog.ENDC}")
    
    save_prediction_details(scenario, "All", pred_len, "Naive_Forecast", 
                            naive_actuals, naive_preds, scaler_target)

    result = {
        "Scenario": scenario, "Lookback": "All", "Horizon": pred_len, "Model": "Naive",
        "Scaled MSE (%)": mse_scaled, "Scaled MAE (%)": mae_scaled, 
        "Real MSE": mse_real, "Real MAE": mae_real 
    }
    
    # ⚙️ MINIMAL CHANGE: Log Naive Baseline to MLflow
    run_name = f"Naive_{scenario.replace(' ', '_')}_H{pred_len}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(MLFLOW_TAGS)
        mlflow.log_params({"Scenario": scenario, "Lookback": "All", "Horizon": pred_len, "Model": "Naive"})
        mlflow.log_metrics({"Test_Scaled_MSE": mse_scaled, "Test_Scaled_MAE": mae_scaled, "Test_Real_MSE": mse_real, "Test_Real_MAE": mae_real})
        
    save_results_to_excel([result], OUTPUT_FILE)
    return result
def perform_hyperparameter_tuning(model_name, train_ds, val_ds, input_dim, pred_len, use_weighted_loss, N_TRIALS, scenario, scaler_target, scaler_cov): # 🚀 Added scaler_cov
    print(f"\n    🚀 Starting Optuna Optimization for: [{model_name}] | Scenario: [{scenario}]")
    
    def objective(trial):
        set_seed(42)
        
        # 🚀 ADDED: Start an MLflow run to log this specific Optuna trial
        with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
            
            # --- MASSIVE TAGGING FOR OPTUNA TRIALS ---
            trial_tags = {
                "source": "Python Script",
                "version": "1.1",
                "models": model_name,
                "description": "Hyperparameter optimization trial",
                "dataset": "AAPL_Multivariate",
                "horizon": pred_len,
                "lookback": train_ds.seq_len,
                "scenario": scenario,
                "trial_number": trial.number,
                "runner": RUNNER_NAME,
                "track": f"{scenario}_H{pred_len}"
            }
            mlflow.set_tags(trial_tags)
            
            lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
            # ⚙️ NEW: Added weight_decay for regularization
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True) 
            dropout = trial.suggest_float('dropout', 0.02, 0.15)
            batch_size = trial.suggest_categorical('batch_size', [4,8, 16, 32])

            if "Informer" in model_name or "wo_Gen" in model_name:
                d_model = trial.suggest_categorical('d_model', [16, 32, 64])
                if d_model == 16: n_heads = 2
                elif d_model == 32: n_heads = 4
                else: n_heads = 4 # for d_model 64
                
                # ⚙️ MINIMAL CHANGE: Extract norm_type dynamically from the model_name string
                n_type = 'revin'
                if 'SANorm' in model_name: n_type = 'san'
                elif 'FANorm' in model_name: n_type = 'fan'
                elif 'WDANorm' in model_name: n_type = 'wdan'
                elif 'CoRANorm' in model_name: n_type = 'cora'
                
                # ⚙️ MINIMAL CHANGE: Passed seq_len and norm_type
                model = Informer(input_dim, seq_len=train_ds.seq_len, d_model=d_model, n_heads=n_heads, out_len=pred_len, 
                                 dropout=dropout, enable_revin=ENABLE_REVIN, norm_type=n_type).to(DEVICE)
                tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            
            elif "PatchTST" in model_name:
                d_model = trial.suggest_categorical('d_model', [16,32,64])
                patch_len = trial.suggest_categorical('patch_len', [8, 16])
                model = PatchTST(input_dim, train_ds.seq_len, pred_len, d_model=d_model, patch_len=patch_len, dropout=dropout, enable_revin=ENABLE_REVIN).to(DEVICE)
                tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
                
            elif "LSTM" in model_name:
                hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
                model = LSTMModel(input_dim, hidden_dim=hidden_dim, out_len=pred_len, dropout=dropout, enable_revin=ENABLE_REVIN).to(DEVICE)
                tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

            elif "TCN" in model_name:
                tcn_kernel = trial.suggest_int('tcn_kernel', 2, 5)
                model = TCNModel(input_dim, output_len=pred_len, kernel_size=tcn_kernel, dropout=dropout, enable_revin=ENABLE_REVIN).to(DEVICE)
                tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

            if use_weighted_loss:
                neg_w, pos_w, neu_w = trial.suggest_float('neg_w', 1.5, 3.5), trial.suggest_float('pos_w', 1.0, 1.8), 0.5
                vol_high_w, direction_w = trial.suggest_float('vol_high_w', 1.0, 3.0), trial.suggest_float('direction_w', 0.5, 2.0)
            else:
                neg_w, pos_w, neu_w, vol_high_w, vol_low_w, direction_w = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

            # 🚀 ADDED: Log the sampled hyperparameters to MLflow
            mlflow.log_params(trial.params)

            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            temp_path = os.path.join(MODEL_SAVE_DIR, f"temp_{model_name}_opt.pth")
            
            # ⚙️ MINIMAL CHANGE: Passed weight_decay down to the training function
            best_mse = train_torch_model(model, tr_loader, val_loader, EPOCHS, lr, PATIENCE, temp_path,
                                         scaler_target=scaler_target,
                                         weight_decay=weight_decay,
                                         use_weighted_loss=use_weighted_loss, enable_custom_loss=ENABLE_CUSTOM_LOSS, 
                                         neg_w=neg_w, pos_w=pos_w, neu_w=neu_w, vol_high_w=vol_high_w, 
                                         direction_w=direction_w, trial=trial, 
                                         adx_idx=train_ds.adx_col_idx, scaler_cov=scaler_cov) # 🚀 FIX 4: Pass scaler
            
            # Evaluate real metrics for the trial to fulfill the requirement
            model.eval()
            t_preds, t_acts = [], []
            with torch.no_grad():
                for b in val_loader:
                    b = [tensor.to(DEVICE) for tensor in b]
                    p = model(b[0], b[1], b[2], b[4], training_target=None) if len(b) == 9 else model(b[0])
                    t_preds.append(p.cpu().numpy())
                    t_acts.append(b[5 if len(b)==9 else 1].cpu().numpy())
            
            t_preds_cat = np.concatenate(t_preds)
            t_acts_cat = np.concatenate(t_acts)
            mse_real_t, mae_real_t, mse_scaled_t, mae_scaled_t = calculate_metrics(t_preds_cat, t_acts_cat, scaler_target)

            # 🚀 ADDED: Log ALL metrics for the trial
            mlflow.log_metrics({
                "trial_best_mse": best_mse,
                "test_scaled_mse": mse_scaled_t,
                "test_scaled_mae": mae_scaled_t,
                "test_real_mse": mse_real_t,
                "test_real_mae": mae_real_t
            })

            if os.path.exists(temp_path): os.remove(temp_path)
            return best_mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params, study.best_trial.user_attrs.get("best_epoch", 50)
class LSTMModel(nn.Module):
    # ⚙️ MINIMAL CHANGE: Added enable_revin=True to the parameters
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, out_len=1, dropout=0.1, enable_revin=True):
        super(LSTMModel, self).__init__()
        
        # ⚙️ MINIMAL CHANGE: Passed the toggle to RevIN
        self.revin = RevIN(input_dim, affine=True, enable_revin=enable_revin)
        
        # We define the LSTM layers based on Optuna/Hyperparam inputs
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.projection = nn.Linear(hidden_dim, out_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, training_target=None):
        # 1. Reversible Instance Normalization (Normalize)
        x, mean, stdev, gamma, beta = self.revin(x_enc, 'norm')
        
        # 2. LSTM Forward Pass
        out, _ = self.lstm(x)
        
        # 3. Take the last hidden state to represent the sequence
        last_out = out[:, -1, :] 
        
        # 4. Project to the prediction horizon
        pred = self.projection(last_out) # (Batch, Out_Len)
        pred = pred.unsqueeze(-1)        # (Batch, Out_Len, 1)
        
        # 5. Denormalize back to MinMaxScaler domain (Target index 0)
        return self.revin(pred, 'denorm', mean=mean, stdev=stdev, gamma=gamma, beta=beta, target_idx=0)
     
# =====================================================================
# MODULAR ABLATION HELPERS
# =====================================================================

def prepare_window_data(lookback, train_data, train_regs, val_data, val_regs, val_overlap_len, train_marks_full, val_marks_full, test_data, test_marks_full, test_regs, test_overlap_len):
    """Slices and aligns chronological data for the current window."""
    full_train_vals = np.concatenate([train_data, val_data[val_overlap_len:]], axis=0)
    full_train_marks = np.concatenate([train_marks_full[:len(train_data)], val_marks_full[val_overlap_len:]], axis=0)
    full_train_regs = np.concatenate([train_regs[:len(train_data)], val_regs[val_overlap_len:]], axis=0)
    
    trim_start = test_overlap_len - lookback
    current_test_data = test_data[trim_start:]
    current_test_marks = test_marks_full[trim_start:]
    current_test_regs = test_regs[trim_start:]
    
    return full_train_vals, full_train_marks, full_train_regs, current_test_data, current_test_marks, current_test_regs


def get_model_datasets(model_name, train_vals, train_marks, train_regs, 
                       val_vals, val_marks, val_regs, 
                       test_vals, test_marks, test_regs, 
                       lookback, pred_len, sent_idx, adx_idx):
    """
    Factory to return the correct Dataset class using strictly separated splits.
    Now includes true GMM Regimes for consistent DRO.
    """
    if any(k in model_name for k in ["PatchTST", "LSTM", "TCN"]):
        train_ds = PatchTSTDataset(train_vals, train_regs, lookback, pred_len, sent_idx, adx_idx)
        val_ds   = PatchTSTDataset(val_vals, val_regs, lookback, pred_len, sent_idx, adx_idx)
        test_ds  = PatchTSTDataset(test_vals, test_regs, lookback, pred_len, sent_idx, adx_idx)
    else:
        train_ds = InformerDataset(train_vals, train_marks, train_regs, lookback, lookback//2, pred_len, sent_idx, adx_idx)
        val_ds   = InformerDataset(val_vals, val_marks, val_regs, lookback, lookback//2, pred_len, sent_idx, adx_idx)
        test_ds  = InformerDataset(test_vals, test_marks, test_regs, lookback, lookback//2, pred_len, sent_idx, adx_idx)
        
    return train_ds, val_ds, test_ds

def generate_xai_attributions(model, model_name, data_loader, feature_cols, device, pred_len, global_train_mean):
    """
    Cutting-edge XAI using Integrated Gradients to map Global Feature Importance.
    Includes critical memory, architecture, and gradient stability checks.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        cprint(f"&nbsp;&nbsp;{ColorLog.WARNING}Captum not installed. Skipping XAI. Run: !pip install captum{ColorLog.ENDC}")
        return None

    ColorLog.info(f"Generating XAI Attributions for {model_name}...")
    
    # Check 1: Ensure Model is strictly in evaluation mode (disables Dropout/BatchNorm shifts)
    model.eval()
    
    # Check 2: Fetch strictly ONE batch to prevent CUDA Out-Of-Memory during intensive gradient calculations
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        return None
        
    batch = [b.to(device) for b in batch]
    enc = batch[0] # Shape: [Batch, SeqLen, Features]
    
    # Check 3: Captum requires the input tensor to track gradients
    enc.requires_grad_()
    
    # 🚀 FIX 5: Dynamic wrapper factory to evaluate SPECIFIC prediction horizons
    def get_forward_wrapper(step_h):
        def forward_wrapper(x):
            # Prevent NaN/Inf gradients from collapsing the integration path
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                
            if any(k in model_name for k in ["PatchTST", "LSTM", "TCN"]):
                out = model(x)
            else:
                enc_mark, dec_input, dec_mark = batch[1], batch[2], batch[4]
                out = model(x, enc_mark, dec_input, dec_mark, training_target=None)
                
            # Return specific horizon step to capture time-specific feature drift
            return out[:, step_h, 0]
        return forward_wrapper

    try:
        # 🚀 FIX 4: Use strictly Zeros for RevIN networks.
        # Since RevIN standardizes input to mean=0 locally, the true baseline is zero.
        # Using the global mean here causes double-shifting scaling artifacts.
        baseline = torch.zeros_like(enc).to(device)
        
        all_step_attributions = []
        all_deltas = [] # Track convergence deltas
        
        for h in range(pred_len):
            ig = IntegratedGradients(get_forward_wrapper(h))
            
            # 🚀 FIX 7: Lock randomness before IG call to prevent ProbSparse stochastic drift
            set_seed(42)
            
            # Calculate attributions (n_steps=50 balances accuracy with memory footprint)
            attributions, delta = ig.attribute(enc, baseline, n_steps=50, return_convergence_delta=True)
            all_step_attributions.append(attributions.detach().cpu().numpy())
            all_deltas.append(delta.detach().cpu().numpy())
            
        # Shape: [Horizon, Batch, SeqLen, Features]
        attributions_np = np.array(all_step_attributions)
        
        # 🚀 FIX 6: Preserve Time Dimension (SeqLen) for deep lag analysis. Average only across Batch.
        # temporal_attr shape: [Horizon, SeqLen, Features]
        temporal_attr = attributions_np.mean(axis=1) 
        
        # Save temporal distribution to disk for paper plots (Regime-conditioned drift)
        temp_file_path = os.path.join(MODEL_SAVE_DIR, f"XAI_Temporal_{model_name}_H{pred_len}.npy")
        np.save(temp_file_path, temporal_attr)
        
        # Compute global feature importance for MLflow by summing absolute lag impacts across time
        # Shape collapses to [Features]
        global_attr = np.abs(temporal_attr).sum(axis=1).mean(axis=0)
        
        # Convert raw gradients into easily readable percentages
        total_attr = np.sum(global_attr) + 1e-8
        importance_pct = (global_attr / total_attr) * 100
        
        xai_results = {}
        for idx, col in enumerate(feature_cols):
            if idx < len(importance_pct): # Safety boundary check
                # Format perfectly for MLflow metric tracking
                clean_col_name = col.replace(" ", "_").replace(".", "")
                xai_results[f"XAI_Importance_{clean_col_name}"] = importance_pct[idx]
        
        # Log convergence delta statistics
        mean_delta = np.mean(all_deltas)
        max_delta = np.max(np.abs(all_deltas))
        xai_results["XAI_Mean_Convergence_Delta"] = mean_delta
        xai_results["XAI_Max_Convergence_Delta"] = max_delta

        # --- NEW: Output XAI to Excel ---
        xai_row = {"Model": model_name, "Horizon": pred_len}
        xai_row.update(xai_results)
        df_xai = pd.DataFrame([xai_row])
        if os.path.exists(XAI_EXCEL_FILE):
            df_xai_exist = pd.read_excel(XAI_EXCEL_FILE)
            pd.concat([df_xai_exist, df_xai], ignore_index=True).to_excel(XAI_EXCEL_FILE, index=False)
        else:
            df_xai.to_excel(XAI_EXCEL_FILE, index=False)

        if mlflow.active_run():
            mlflow.log_metrics(xai_results)
            # Upload the preserved 3D array directly to DagsHub
            mlflow.log_artifact(temp_file_path)
            
        top_feature_idx = np.argmax(importance_pct)
        cprint(f"&nbsp;&nbsp;{ColorLog.OKGREEN}XAI Complete! Highest driving feature: {feature_cols[top_feature_idx]} ({importance_pct[top_feature_idx]:.2f}%){ColorLog.ENDC}")
        
        # Clean up CUDA memory manually post-attribution
        del all_step_attributions, attributions_np, temporal_attr, baseline, enc, batch
        torch.cuda.empty_cache()
        
        return xai_results
        
    except Exception as e:
        cprint(f"&nbsp;&nbsp;{ColorLog.WARNING}XAI Computation failed for {model_name}: {str(e)}{ColorLog.ENDC}")
        torch.cuda.empty_cache()
        return None
def process_standalone_models(scenario, lookback, pred_len, full_train_vals, full_train_marks, full_train_regs, val_data, val_marks, val_regs, current_test_data, current_test_marks, current_test_regs, input_dim, current_sent_idx, current_adx_idx, use_weighted_loss, scaler_target, standalone_configs, feature_cols, scaler_cov): 
    """Phase 1: Tunes, builds, trains, and evaluates all standalone models with embedding ablations."""
    results, trained_cache = [], {} 
    inf_params_cache, inf_epoch_cache = None, None
    patch_params_cache, patch_epoch_cache = None, None # NEW: Cache for PatchTST

    ColorLog.extreme(f"PHASE 1: TUNING & MC-TRAINING | L:{lookback} | H:{pred_len}", color="#00CED1", border_color="#00CED1")
    
    for model_name, config in standalone_configs.items():
        if is_completed(scenario, lookback, pred_len, model_name):
            ColorLog.info(f"⏭️ Checkpoint found. Skipping already completed model: {model_name}")
            continue

        ColorLog.extreme(f"NOW PROCESSING: {model_name}", color="#FFA500", border_color="#FFA500")
        
        # 1. Dataset Generation
        train_ds, val_ds, test_ds = get_model_datasets(model_name, full_train_vals, full_train_marks, full_train_regs,
                                                       val_data, val_marks, val_regs, 
                                                       current_test_data, current_test_marks, current_test_regs,
                                                       lookback, pred_len, current_sent_idx, current_adx_idx)
        
        # 2. Hyperparameter Tuning (with Parameter Caching)
        is_informer_type = any(k in model_name for k in ["Informer", "wo_Gen"])
        is_patch_type = config.get("is_patch", False)
        
        if is_informer_type and inf_params_cache is not None:
            best_params, opt_epoch = inf_params_cache, inf_epoch_cache
            ColorLog.extreme(f"CACHE HIT! REUSING TUNED PARAMS FOR {model_name}", color="#00FFFF", border_color="#00FFFF")
        elif is_patch_type and patch_params_cache is not None:
            best_params, opt_epoch = patch_params_cache, patch_epoch_cache
            ColorLog.extreme(f"CACHE HIT! REUSING TUNED PARAMS FOR {model_name}", color="#00FFFF", border_color="#00FFFF")
        else:
            # FIX: Use actual model_name for tuning track, don't override to standalone
            tune_name = model_name 
            best_params, opt_epoch = perform_hyperparameter_tuning(tune_name, train_ds, val_ds, input_dim, pred_len, 
                                                                  use_weighted_loss, N_TRIALS, scenario, scaler_target, scaler_cov)
            if is_informer_type: 
                inf_params_cache, inf_epoch_cache = best_params, opt_epoch
            if model_name == "PatchTST_Standalone":
                patch_params_cache, patch_epoch_cache = best_params, opt_epoch

        # 3. Architecture Construction
        if config.get("is_patch"):
            model = PatchTST(input_dim, lookback, pred_len, d_model=best_params['d_model'], 
                             patch_len=best_params.get('patch_len', 16), e_layers=best_params.get('e_layers', 3), 
                             enable_revin=ENABLE_REVIN).to(DEVICE)
        elif config.get("is_lstm"):
            model = LSTMModel(input_dim, hidden_dim=best_params.get('hidden_dim', 64), 
                              num_layers=2, out_len=pred_len, enable_revin=ENABLE_REVIN).to(DEVICE)
        elif config.get("is_tcn"):
            model = TCNModel(input_dim, output_len=pred_len, num_channels=[64, 64], 
                             kernel_size=best_params.get('tcn_kernel', 2), enable_revin=ENABLE_REVIN).to(DEVICE)
        else:
            d_m = best_params['d_model']
            n_h = 2 if d_m == 16 else 4 if d_m <= 64 else 8
            
            model = Informer(input_dim, seq_len=lookback, d_model=d_m, n_heads=n_h, out_len=pred_len, 
                             distil=config.get('distil', True), prob_sparse=config.get('prob_sparse', True), 
                             dec_only=config.get('dec_only', False), gen_dec=config.get('gen_dec', True), 
                             dropout=best_params['dropout'], enable_revin=ENABLE_REVIN,
                             use_temporal=config.get('use_temporal', True),
                             use_positional=config.get('use_positional', True),
                             norm_type=config.get('norm_type', 'revin')).to(DEVICE)

        run_name = f"{model_name}_{scenario.replace(' ', '_')}_L{lookback}_H{pred_len}"
        with mlflow.start_run(run_name=run_name):
            is_tta = config.get('tta', False)
            norm_used = config.get('norm_type', 'revin' if ENABLE_REVIN else 'none')
            
            # --- MASSIVE TAGGING REQUIREMENT ---
            full_tags = {
                "source": "Python Pipeline",
                "version": "1.1",
                "models": model_name,
                "description": "Primary Model Evaluation Run",
                "dataset": "AAPL_Processed",
                "horizon": pred_len,
                "lookback": lookback,
                "model": model_name,
                "scenario": scenario,
                "tta": is_tta,
                "batch_size": best_params.get('batch_size', 'N/A'),
                "dropout": best_params.get('dropout', 'N/A'),
                "hidden_dim": best_params.get('hidden_dim', best_params.get('d_model', 'N/A')),
                "lr": best_params.get('lr', 'N/A'),
                "neg_w": best_params.get('neg_w', 'N/A'),
                "pos_w": best_params.get('pos_w', 'N/A'),
                "tcn_kernel": best_params.get('tcn_kernel', 'N/A'),
                "vol_high_w": best_params.get('vol_high_w', 'N/A'),
                "weight_decay": best_params.get('weight_decay', 'N/A'),
                "runner": RUNNER_NAME,
                "track": f"{scenario}_H{pred_len}",
                "normalization_used": norm_used
            }
            # Remove N/A to keep tags clean
            full_tags = {k: v for k, v in full_tags.items() if v != 'N/A'}
            mlflow.set_tags(full_tags)
            
            mlflow.log_params({"Scenario": scenario, "Lookback": lookback, "Horizon": pred_len, "Model": model_name, "TTA": is_tta})
            mlflow.log_params(best_params) 

            # 4. Training Phase
            model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pth")
            
            base_model_name = model_name.replace("_TTA", "")
            if is_tta and base_model_name in trained_cache:
                ColorLog.extreme(f"LOADING BASE WEIGHTS FOR TTA FROM {base_model_name}", color="#FF1493", border_color="#FF1493")
                model.load_state_dict(trained_cache[base_model_name].state_dict())
            else:
                curr_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=False, drop_last=True)
                
                if config.get("is_patch") and config.get("pretrain"):
                    pretrain_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_pretrained.pth")
                    pretrain_patchtst_model(model, curr_loader, None, max(1, opt_epoch), best_params['lr'], PATIENCE, pretrain_path)

                # 🚀 MINIMAL CHANGE: Pass 'dro' flag from config and explicit adx_idx to the training function
                train_torch_model(model, curr_loader, None, max(1, opt_epoch), best_params['lr'], PATIENCE, model_path,
                                  scaler_target=scaler_target, use_weighted_loss=use_weighted_loss, 
                                  enable_custom_loss=ENABLE_CUSTOM_LOSS, trial=None, use_dro=config.get('dro', False),
                                  adx_idx=current_adx_idx, scaler_cov=scaler_cov, # 🚀 FIX 4: Pass scaler
                                  **{k: best_params[k] for k in best_params if k in ['neg_w', 'pos_w', 'neu_w', 'vol_high_w', 'direction_w']})
            
            trained_cache[model_name] = model

            # 5. Evaluation with MC-Dropout and Test-Time Adaptation (TTA)
            # ⚙️ TTA requires strict chronological tracking (batch_size=1) to prevent future leakage
            bs = 1 if is_tta else best_params['batch_size']
            te_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            
            batch_preds = []
            all_acts = []

            # 🚀 FIX 2: Initialize TTA model OUTSIDE the loop to carry context forward
            if is_tta:
                tta_model = copy.deepcopy(model)
                tta_model.train()
                for name, param in tta_model.named_parameters():
                    if 'revin' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, tta_model.parameters()), lr=1e-3)
                eval_model = tta_model
            else:
                eval_model = model
                eval_model.train() # Keep dropout active for standard MC

            # 🔴 FIX (Issue D): Removed strict `set_seed(42)` inside the test block.
            # By allowing the global RNG to progress naturally, we ensure ProbSparse 
            # queries and MC-Dropout masks both maintain true Monte Carlo independence 
            # across different test batches, ensuring optimal uncertainty calibration.

            for batch in te_loader:
                batch = [b.to(DEVICE) for b in batch]
                
                # 🚀 STEP 1: Predict Next Step FIRST (Zero Future Leakage)
                mc_samples = []
                
                with torch.no_grad():
                    for i in range(10): 
                        # Randomness now comes from the model's evolved state (Dropout + ProbSparse)
                        if any(k in model_name for k in ["PatchTST", "LSTM", "TCN"]):
                            p = eval_model(batch[0])
                        else:
                            p = eval_model(batch[0], batch[1], batch[2], batch[4])
                        mc_samples.append(p.cpu().numpy())
                        
                batch_preds.append(np.stack(mc_samples))
                all_acts.append(batch[-3].cpu().numpy()) # Dynamically retrieves 'y' for any dataset format
                
                # 🚀 STEP 2: Adapt ONLY AFTER Prediction using your Self-Supervised Task
                if is_tta:
                    # 3. Self-Supervised TTA Task: Context Shift Forecasting (Intact!)
                    # 🚀 FIX: Remove artificial stationary bias by rolling and zero-padding instead of duplicating
                    shifted_enc = torch.roll(batch[0], shifts=pred_len, dims=1)
                    shifted_enc[:, :pred_len, :] = 0.0
                    target_y = batch[0][:, -pred_len:, 0:1] 
                    
                    optimizer.zero_grad()
                    if any(k in model_name for k in ["PatchTST", "LSTM", "TCN"]):
                        tta_pred = tta_model(shifted_enc)
                    else:
                        # 🚀 MINIMAL CHANGE: Shift Decoder inputs and Time Marks to match the shifted context
                        shifted_dec = torch.roll(batch[2], shifts=pred_len, dims=1)
                        shifted_dec[:, :pred_len, :] = 0.0
                        
                        shifted_enc_mark = torch.roll(batch[1], shifts=pred_len, dims=1)
                        shifted_enc_mark[:, :pred_len, :] = 0.0 # Positional/time embeddings can handle zeros
                        
                        shifted_dec_mark = torch.roll(batch[4], shifts=pred_len, dims=1)
                        shifted_dec_mark[:, :pred_len, :] = 0.0
                        
                        tta_pred = tta_model(shifted_enc, shifted_enc_mark, shifted_dec, shifted_dec_mark)
                        
                    loss = nn.MSELoss()(tta_pred, target_y)
                    loss.backward()
                    optimizer.step()
            
            # Aggregate all predictions and actuals
            mc_stack = np.concatenate(batch_preds, axis=1)
            all_p = mc_stack.mean(axis=0)
            all_a = np.concatenate(all_acts)
            
            mse_real, mae_real, mse_scaled, mae_scaled = calculate_metrics(all_p, all_a, scaler_target)
            
            # 7️⃣ Prediction Confidence Calibration (PICP)
            uncertainty = mc_stack.std(axis=0).mean()
            lower_bound = np.percentile(mc_stack, 2.5, axis=0) # 95% Confidence Interval
            upper_bound = np.percentile(mc_stack, 97.5, axis=0)
            picp = ((all_a >= lower_bound) & (all_a <= upper_bound)).mean() 
            
            # 6️⃣ Model Complexity
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 🚀 FIX: Calculate global training mean for XAI neutral baseline consistency
            global_train_mean = torch.tensor(train_ds.data.mean(axis=0), dtype=torch.float32).to(DEVICE)
            xai_dict = generate_xai_attributions(eval_model, model_name, te_loader, feature_cols, DEVICE, pred_len, global_train_mean)
            
            # NEW: Plot Global Trajectory and Error Distribution
            DeepLearningVisualizer.plot_and_log_global_trajectory(
                all_actuals=all_a, all_preds=all_p, model_name=model_name, 
                horizon=pred_len, lookback=lookback
            )
            DeepLearningVisualizer.plot_and_log_error_distribution(
                actuals=all_a, predictions=all_p, model_name=model_name, horizon=pred_len
            )
            
            # NEW: Plot Uncertainty Fan Chart (using the first batch sequence)
            DeepLearningVisualizer.plot_and_log_uncertainty_fan(
                actual_seq=all_a[0, :, 0], pred_mean_seq=all_p[0, :, 0], 
                lower_bound=lower_bound[0, :, 0], upper_bound=upper_bound[0, :, 0], 
                model_name=model_name, horizon=pred_len
            )
            
            temp_npy_path = os.path.join(MODEL_SAVE_DIR, f"XAI_Temporal_{model_name}_H{pred_len}.npy")
            DeepLearningVisualizer.plot_and_log_xai_heatmap(
                npy_filepath=temp_npy_path, feature_cols=feature_cols, 
                model_name=model_name, horizon=pred_len
            )
            
            ColorLog.success(f"{model_name:<25} | MSE: {mse_scaled:.6f}% | PICP: {picp:.4f} | Params: {param_count:,}")

            res = {"Scenario": scenario, "Lookback": lookback, "Horizon": pred_len, "Model": model_name,
                   "Scaled MSE (%)": mse_scaled, "Scaled MAE (%)": mae_scaled,
                   "Real MSE": mse_real, "Real MAE": mae_real, 
                   "Uncertainty": uncertainty, "PICP": picp, "Param_Count": param_count, "Best Params": str(best_params)}
            
            mlflow.log_metric("Param_Count", param_count)
            mlflow.log_metrics({"Test_Scaled_MSE": mse_scaled, "Test_Scaled_MAE": mae_scaled, "Test_Real_MSE": mse_real, "Test_Real_MAE": mae_real, "Uncertainty": uncertainty, "PICP": picp})
                
        # Write to excel after the run finishes tracking
        save_results_to_excel([res], OUTPUT_FILE)
        results.append(res)
        mark_completed(scenario, lookback, pred_len, model_name) # Mark Checkpoint
        
    return results, trained_cache

def run_ablation_retraining(scenario, lookback, pred_len, train_data, train_regs, val_data, val_regs, val_overlap_len,
                            train_marks_full, val_marks_full, test_data, test_marks_full, test_regs,
                            input_dim, current_sent_idx, current_adx_idx, 
                            use_weighted_loss, scaler_target, test_overlap_len, feature_cols, scaler_cov): 
    
    # 1. Prepare Data Alignment
    full_train_vals, full_train_marks, full_train_regs, current_test_data, current_test_marks, current_test_regs = prepare_window_data(
        lookback, train_data, train_regs, val_data, val_regs, val_overlap_len, 
        train_marks_full, val_marks_full, test_data, test_marks_full, test_regs, test_overlap_len
    )

    # 2. Define Ablation Suite (⚙️ MINIMAL CHANGE: Added Informer_DRO)
    standalone_configs = {
        "Informer_Base": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "revin"},
        "Informer_Base_TTA": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "revin", "tta": True},
        "Informer_DRO": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "revin", "dro": True}, # 🚀 NEW: Robust Optimization
        "Informer_SANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "san"},
        "Informer_FANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "fan"},
        "Informer_FANorm_TTA": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "fan", "tta": True},
        "Informer_WDANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "wdan"},
        "Informer_CoRANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "cora"},
        "Informer_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "use_temporal": False, "use_positional": True},
        "Informer_NoPositional": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "use_temporal": True, "use_positional": False},
        "Informer_NoBothEmbeds": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "use_temporal": False, "use_positional": False},
        "Informer_DecOnly": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True},
        "Informer_FullAttn": {"distil": True, "prob_sparse": False, "dec_only": False, "gen_dec": True},
        "Informer_NoDistil": {"distil": False, "prob_sparse": True, "dec_only": False, "gen_dec": True},
        "Informer_NoProb_NoDistil": {"distil": False, "prob_sparse": False, "dec_only": False, "gen_dec": True},
        "wo_Gen_Dec": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": False},
        "Informer_CoRANNorm_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "cora", "use_temporal": False, "use_positional": True},
        "Informer_DecOnly_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True, "use_temporal": False, "use_positional": True},
        "Informer_CoRanNorm_DecOnly": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True, "norm_type": "cora"},
        "Informer_CoRanNorm_DecOnly_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True, "norm_type": "cora", "use_temporal": False, "use_positional": True},
        "PatchTST_Standalone": {"is_patch": True, "pretrain": False},
        "PatchTST_Pretrained": {"is_patch": True, "pretrain": True},
        "LSTM_Standalone": {"is_lstm": True},
        "TCN_Standalone": {"is_tcn": True}
    }

    # 3. Phase 1: Process Standalone Models (Tuning -> Training -> MC-Dropout)
    results, trained_cache = process_standalone_models(
        scenario, lookback, pred_len, full_train_vals, full_train_marks, full_train_regs,
        val_data, val_marks_full, val_regs, 
        current_test_data, current_test_marks, current_test_regs, input_dim, 
        current_sent_idx, current_adx_idx, use_weighted_loss, scaler_target, standalone_configs, feature_cols, scaler_cov 
    )

    # 4. Phase 2: Dynamic Ensembles (Inverse-MSE Weighting including Naive)
    ensemble_results = evaluate_ensembles(
        scenario, lookback, pred_len, results, trained_cache, 
        full_train_vals, full_train_marks, full_train_regs, # 🚀 FIX: Pass full training sets for overlap
        val_data, val_marks_full, val_regs, 
        current_test_data, current_test_marks, current_test_regs,
        current_sent_idx, current_adx_idx, scaler_target
    )
    
    # 5. Final Aggregation & Save Champion weights for this window
    results.extend(ensemble_results)
    save_overall_best_model(scenario, lookback, pred_len, results, trained_cache)

    return results


def evaluate_ensembles(scenario, lookback, pred_len, results, trained_cache, 
                       train_data, train_marks, train_regs, val_data, val_marks, val_regs, # 🚀 FIX: Clean splits + Regimes
                       current_test_data, current_test_marks, current_test_regs,
                       current_sent_idx, current_adx_idx, scaler_target):
    """Phase 2: Computes Dynamic Inverse-MSE weights for ALL models, including Naive."""
    
    # 1. Identify winning variants from standalone runs
    inf_results = [r for r in results if "Informer" in r['Model'] or "wo_Gen" in r['Model']]
    best_inf_name = min(inf_results, key=lambda x: x['Scaled MSE (%)'])['Model']
    
    patch_results = [r for r in results if "PatchTST" in r['Model']]
    best_patch_name = min(patch_results, key=lambda x: x['Scaled MSE (%)'])['Model']
    
    ColorLog.extreme(f"PHASE 2: DYNAMIC ENSEMBLE EVALUATION | L:{lookback} H:{pred_len}", color="#00FF00", border_color="#00FF00")

    # 🚀 FIX: Create overlap-aware validation sets
    overlap_val_data = np.concatenate([train_data[-lookback:], val_data], axis=0)
    overlap_val_marks = np.concatenate([train_marks[-lookback:], val_marks], axis=0)
    overlap_val_regs = np.concatenate([train_regs[-lookback:], val_regs], axis=0)

    # 2. Setup Validation Loaders (🚀 FIX: Use overlap sets and include regimes)
    val_ds_inf = InformerDataset(overlap_val_data, overlap_val_marks, overlap_val_regs, lookback, lookback//2, pred_len, current_sent_idx, current_adx_idx)
    val_loader_inf = DataLoader(val_ds_inf, batch_size=32, shuffle=False)
    
    val_ds_patch = PatchTSTDataset(overlap_val_data, overlap_val_regs, lookback, pred_len, current_sent_idx, current_adx_idx)
    val_loader_patch = DataLoader(val_ds_patch, batch_size=32, shuffle=False)
    
    # 3. Calculate Validation MSE for each Deep Learning model
    val_mses = {}
    dl_models = [best_inf_name, best_patch_name, "LSTM_Standalone", "TCN_Standalone"]
    
    for m_name in dl_models:
        model = trained_cache[m_name].eval()
        v_preds, v_acts = [], []
        loader = val_loader_inf if any(k in m_name for k in ["Informer", "wo_Gen"]) else val_loader_patch
        with torch.no_grad():
            for batch in loader:
                batch = [b.to(DEVICE) for b in batch]
                # 🚀 FIX: Disable Teacher Forcing (training_target=None) for honest ensemble weights
                p = model(batch[0], batch[1], batch[2], batch[4], training_target=None) if any(k in m_name for k in ["Informer", "wo_Gen"]) else model(batch[0])
                v_preds.append(p.cpu().numpy()); v_acts.append(batch[5 if len(batch)==9 else 1].cpu().numpy())
        
        # Standardize calculation via calculate_metrics
        _, _, v_mse_scaled, _ = calculate_metrics(np.concatenate(v_preds), np.concatenate(v_acts), scaler_target)
        val_mses[m_name] = v_mse_scaled

    # 4. Standardized Naive Validation MSE (Dynamic Weighting Logic)
    v_preds_naive, v_acts_naive = [], []
    for batch in val_loader_patch:
        last_known_val = batch[0][:, -1:, 0:1] 
        p_naive = last_known_val.repeat(1, pred_len, 1)
        v_preds_naive.append(p_naive.numpy()); v_acts_naive.append(batch[1].numpy())
    
    _, _, naive_val_mse, _ = calculate_metrics(np.concatenate(v_preds_naive), np.concatenate(v_acts_naive), scaler_target)
    val_mses['Naive'] = naive_val_mse

    # 5. Compute Dynamic Weights (Inverse-MSE for all 5 components)
    all_names = dl_models + ['Naive']
    inv_mses = [1.0 / (val_mses[m] + 1e-9) for m in all_names]
    total_inv = sum(inv_mses)
    dynamic_weights = [inv / total_inv for inv in inv_mses]
    
    cprint(f"{ColorLog.OKBLUE}[DYNAMIC WEIGHTS]{ColorLog.ENDC} {dict(zip(all_names, [round(w, 6) for w in dynamic_weights]))}")
    
    # 📊 VISUALIZATION HOOK 2: Dynamic Ensemble Weights
    DeepLearningVisualizer.plot_and_log_ensemble_weights(
        model_names=all_names, weights=dynamic_weights, 
        ensemble_name="Ensemble_Dynamic_InvMSE", 
        horizon=pred_len, lookback=lookback
    )
    
    ensemble_configs = {
        "Ensemble_Equal_Weight": [0.20] * 5, 
        "Ensemble_Dynamic_InvMSE": dynamic_weights
    }

    ensemble_results = []
    # 🚀 FIX: Add current_test_regs to the Test InformerDataset
    test_ds_inf = InformerDataset(current_test_data, current_test_marks, current_test_regs, lookback, lookback//2, pred_len, current_sent_idx, current_adx_idx)
    te_loader_inf = DataLoader(test_ds_inf, batch_size=32, shuffle=False)
    
    models_in_ensemble = f"{best_inf_name}, {best_patch_name}, LSTM_Standalone, TCN_Standalone, Naive"

    for model_name, weights in ensemble_configs.items():
        if is_completed(scenario, lookback, pred_len, model_name):
            ColorLog.info(f"⏭️ Checkpoint found. Skipping Ensemble: {model_name}")
            continue
        model = EnsembleModel(trained_cache[best_inf_name], trained_cache[best_patch_name], 
                              trained_cache["LSTM_Standalone"], trained_cache["TCN_Standalone"], weights=weights).to(DEVICE)
        trained_cache[model_name] = model.eval()
        
        preds, actuals = [], []
        with torch.no_grad():
            for batch in te_loader_inf:
                batch = [b.to(DEVICE) for b in batch]
                # 🚀 Test evaluation always uses None for training_target
                p = model(batch[0], batch[1], batch[2], batch[4], training_target=None)
                preds.append(p.cpu().numpy()); actuals.append(batch[5].cpu().numpy())

        all_p, all_a = np.concatenate(preds), np.concatenate(actuals)
        save_prediction_details(scenario, lookback, pred_len, model_name, all_a, all_p, scaler_target)
        
        mse_real, mae_real, mse_scaled, mae_scaled = calculate_metrics(all_p, all_a, scaler_target)
        
        # 📊 VISUALIZATION HOOK 3: Ensemble Forecast Trajectories
        DeepLearningVisualizer.plot_and_log_forecast(
            actuals=all_a, predictions=all_p, model_name=model_name, 
            horizon=pred_len, lookback=lookback
        )
        
        ColorLog.success(f"🏆 [{model_name:<25}] Scaled MSE (%): {mse_scaled:.6f}%")
        
        res = {"Scenario": scenario, "Lookback": lookback, "Horizon": pred_len, "Model": model_name,
               "Scaled MSE (%)": mse_scaled, "Scaled MAE (%)": mae_scaled,
               "Real MSE": mse_real, "Real MAE": mae_real, 
               "Best Params": f"Weights: {[round(float(w), 4) for w in weights]}"}
               
        run_name = f"{model_name}_{scenario.replace(' ', '_')}_L{lookback}_H{pred_len}"
        with mlflow.start_run(run_name=run_name):
            ens_tags = MLFLOW_TAGS.copy()
            ens_tags.update({
                "models_selected_in_ensemble": models_in_ensemble,
                "model": model_name,
                "horizon": pred_len,
                "lookback": lookback,
                "scenario": scenario
            })
            mlflow.set_tags(ens_tags)
            mlflow.log_params({"Scenario": scenario, "Lookback": lookback, "Horizon": pred_len, "Model": model_name})
            mlflow.log_params({"Weight_Inf": weights[0], "Weight_PT": weights[1], "Weight_LSTM": weights[2], "Weight_TCN": weights[3], "Weight_Naive": weights[4]})
            mlflow.log_metrics({"Test_Scaled_MSE": mse_scaled, "Test_Scaled_MAE": mae_scaled, "Test_Real_MSE": mse_real, "Test_Real_MAE": mae_real})
            
        save_results_to_excel([res], OUTPUT_FILE)
        ensemble_results.append(res)
        mark_completed(scenario, lookback, pred_len, model_name) # Mark Checkpoint

    return ensemble_results


def save_overall_best_model(scenario, lookback, pred_len, all_results_for_window, trained_cache):
    """Scans all results for the window and saves the weights and object of the #1 performer."""
    # ⚙️ MINIMAL CHANGE: Filter results to only include trained PyTorch models (excluding 'Naive')
    saveable_results = [r for r in all_results_for_window if r['Model'] in trained_cache]
    
    if not saveable_results:
        return

    best_overall_run = min(saveable_results, key=lambda x: x['Scaled MSE (%)'])
    best_overall_name = best_overall_run['Model']
    best_overall_model = trained_cache[best_overall_name]
    
    clean_scenario = scenario.replace(" ", "_")
    weights_save_path = os.path.join(MODEL_SAVE_DIR, f"Overall_Best_Weights_{clean_scenario}_L{lookback}_H{pred_len}.pth")
    full_model_save_path = os.path.join(MODEL_SAVE_DIR, f"Overall_Best_FullModel_{clean_scenario}_L{lookback}_H{pred_len}.pth")
    
    torch.save(best_overall_model.state_dict(), weights_save_path)
    torch.save(best_overall_model, full_model_save_path)
    
    # ⚙️ MINIMAL CHANGE: Log the physical best model to MLflow as an artifact
    run_name = f"CHAMPION_{best_overall_name}_{clean_scenario}_L{lookback}_H{pred_len}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(MLFLOW_TAGS)
        mlflow.log_params({"Scenario": scenario, "Lookback": lookback, "Horizon": pred_len, "Champion_Model": best_overall_name})
        mlflow.log_metric("Champion_Scaled_MSE", best_overall_run['Scaled MSE (%)'])
        
        # Log the PyTorch model artifact AND register it in the MLflow Model Registry UI
        registry_name = f"Champion_{clean_scenario}_H{pred_len}"
        mlflow.pytorch.log_model(
            pytorch_model=best_overall_model, 
            artifact_path="model", 
            registered_model_name=registry_name
        )
    
    # --- NEW: Output to separate Summary Excel sheet ---
    summary_data = {
        "Scenario": scenario,
        "Lookback": lookback,
        "Horizon": pred_len,
        "Best_Model": best_overall_name,
        "test_scaled_mse": best_overall_run.get('Scaled MSE (%)'),
        "test_scaled_mae": best_overall_run.get('Scaled MAE (%)'),
        "real_mse": best_overall_run.get('Real MSE'),
        "real_mae": best_overall_run.get('Real MAE'),
        "parameters": best_overall_run.get('Best Params', 'N/A'),
        "epoch_count": best_overall_run.get('Epochs', 'Cached/Optuna'),
        "models_used_if_ensemble": best_overall_run.get('Best Params') if "Ensemble" in best_overall_name else "N/A"
    }
    
    # Check for Naive explicitly
    naive_run = next((r for r in all_results_for_window if r['Model'] == 'Naive'), None)
    if naive_run:
        summary_data.update({
            "Naive_test_scaled_mse": naive_run.get('Scaled MSE (%)'),
            "Naive_test_scaled_mae": naive_run.get('Scaled MAE (%)'),
            "Naive_real_mse": naive_run.get('Real MSE'),
            "Naive_real_mae": naive_run.get('Real MAE')
        })

    df_summary = pd.DataFrame([summary_data])
    if os.path.exists(SUMMARY_EXCEL_FILE):
        df_exist = pd.read_excel(SUMMARY_EXCEL_FILE)
        pd.concat([df_exist, df_summary], ignore_index=True).to_excel(SUMMARY_EXCEL_FILE, index=False)
    else:
        df_summary.to_excel(SUMMARY_EXCEL_FILE, index=False)

    mse_val = best_overall_run['Scaled MSE (%)']
    ColorLog.success(f"💾 SAVED OVERALL BEST MODEL: [{best_overall_name}] (MSE: {mse_val:.6f}%) for {scenario} (L:{lookback}, H:{pred_len})")
def get_best_model_configs(results_path="./Paper_Replication_Results.xlsx"):
    """
    Analyzes the replication results to find the overall best model
    and the specific Informer ablation winner.
    """
    if not os.path.exists(results_path):
        print("Results file not found. Run the training pipeline first.")
        return None

    df = pd.read_excel(results_path)
    
    # 1. Find the Absolute Best Model (Comparing all 5: Inf, PT, TCN, LSTM, Naive)
    # We use Scaled MSE (%) for the final decision to match the paper
    best_overall = df.loc[df['Scaled MSE (%)'].idxmin()]
    
    print("="*30)
    print("OVERALL ACCURACY LEADER")
    print(f"Model: {best_overall['Model']}")
    print(f"Scenario: {best_overall['Scenario']}")
    print(f"Horizon: {best_overall['Horizon']} | Lookback: {best_overall['Lookback']}")
    # ⚙️ MINIMAL CHANGE: Updated formatting from .2f to .6f
    print(f"Scaled MSE (%): {best_overall['Scaled MSE (%)']:.6f}%")
    print("="*30)

    # 2. Select the Best Informer Variant from Ablation
    # Filter for Informer variants and the 'wo_Gen_Dec' variant
    informer_variants = df[df['Model'].str.contains('Informer|wo_Gen', case=False, na=False)]
    
    if informer_variants.empty:
        print("No Informer ablation results found.")
        return best_overall, None

    best_inf_row = informer_variants.loc[informer_variants['Scaled MSE (%)'].idxmin()]
    
    # --- CRITICAL CHANGE: Updated to include Normalization and Embedding ablations ---
    ablation_map = {
        "Informer_Base": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "revin"},
        "Informer_SANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "san"},
        "Informer_FANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "fan"},
        "Informer_WDANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "wdan"},
        "Informer_CoRANorm": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "cora"},
        "Informer_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "use_temporal": False, "use_positional": True},
        "Informer_NoPositional": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "use_temporal": True, "use_positional": False},
        "Informer_NoBothEmbeds": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "use_temporal": False, "use_positional": False},
        "Informer_DecOnly": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True},
        "Informer_FullAttn": {"distil": True, "prob_sparse": False, "dec_only": False, "gen_dec": True},
        "Informer_NoDistil": {"distil": False, "prob_sparse": True, "dec_only": False, "gen_dec": True},
        "Informer_NoProb_NoDistil": {"distil": False, "prob_sparse": False, "dec_only": False, "gen_dec": True},
        "wo_Gen_Dec": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": False},
        "Informer_CoRANNorm_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": False, "gen_dec": True, "norm_type": "cora", "use_temporal": False, "use_positional": True},
        "Informer_DecOnly_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True, "use_temporal": False, "use_positional": True},
        "Informer_CoRanNorm_DecOnly": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True, "norm_type": "cora"},
        "Informer_CoRanNorm_DecOnly_NoTemporal": {"distil": True, "prob_sparse": True, "dec_only": True, "gen_dec": True, "norm_type": "cora", "use_temporal": False, "use_positional": True}
    }

    best_inf_name = best_inf_row['Model']
    best_inf_config = ablation_map.get(best_inf_name, ablation_map["Informer_Base"])
    
    print("\nINFORMER ABLATION WINNER")
    print(f"Best Variant: {best_inf_name}")
    print(f"Config: {best_inf_config}")
    # ⚙️ MINIMAL CHANGE: Updated formatting from .2f to .6f
    print(f"Scaled MSE (%): {best_inf_row['Scaled MSE (%)']:.6f}%")
    
    return best_overall, best_inf_config
class EnsembleModel(nn.Module):
    def __init__(self, informer, patchtst, lstm, tcn, weights=[0.25, 0.25, 0.2, 0.2, 0.1]):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleDict({'inf': informer, 'pt': patchtst, 'lstm': lstm, 'tcn': tcn})
        self.w = weights 
        self.pred_len = informer.pred_len
        
        # 🔴 FIX (Issue E): Architecturally lock submodels to prevent gradient flow
        # ensuring absolute safety if the ensemble is ever accidentally trained jointly.
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, enc, enc_mark, dec, dec_mark, training_target=None):
        p_inf = self.models['inf'](enc, enc_mark, dec, dec_mark, training_target)
        p_pt  = self.models['pt'](enc)
        p_lstm = self.models['lstm'](enc)
        p_tcn  = self.models['tcn'](enc)

        # CRITICAL FIX: The models output directly in the MinMaxScaler domain.
        # The naive forecast can safely use the last scaled input directly.
        with torch.no_grad():
            last_scaled_val = enc[:, -1:, 0:1] 
        
        p_naive = last_scaled_val.repeat(1, self.pred_len, 1)

        out = (self.w[0] * p_inf + 
               self.w[1] * p_pt + 
               self.w[2] * p_lstm + 
               self.w[3] * p_tcn + 
               self.w[4] * p_naive)
        
        return out
        
def main():
    # ⚙️ Ensure clean MLflow state before starting
    if mlflow.active_run():
        mlflow.end_run()
    
    # ⚙️ Determine total dataset length
    if not os.path.exists(DATA_PATH):
        total_length = 2200 
    else:
        total_length = len(pd.read_csv(DATA_PATH))
        
    prediction_horizons = [TARGET_HORIZON]
    lookback_windows = [96, 32, 16]
    scenarios = [TARGET_SCENARIO]
    max_lookback_needed = max(lookback_windows)

    ColorLog.info("Mode: Static Train/Val/Test Split Enabled")
    ColorLog.section("STARTING EVALUATION")
    
    ColorLog.extreme("PIPELINE INITIALIZED", color="#FFFFFF", border_color="#FFFFFF")

    # 📊 Results accumulator
    all_metrics = []

    ColorLog.extreme("LOADING & SPLITTING DATASET", color="#FFD700", border_color="#FFD700")
    # Load Window Data (Slices are now safely computed inside the function)
    
    # We must quickly peek at the columns to get feature names for XAI
    df_temp = pd.read_csv(DATA_PATH, nrows=0) if os.path.exists(DATA_PATH) else pd.DataFrame(columns=['Close', 'RSI', 'SMA', 'ADX', 'Gold', 'USDJPY', 'Sentiment_Score'])
    raw_cols = [c for c in df_temp.columns if c not in ['Date', 'Regime']]
    if 'Close' in raw_cols: raw_cols.insert(0, raw_cols.pop(raw_cols.index('Close')))
    
    # 🚀 NEW: Unpack the GMM Regimes
    (train_vals_full, train_marks_full, train_regs_full,
     val_vals_full, val_marks_full, val_regs_full,
     test_vals_full, test_marks_full, test_regs_full,
     input_dim_full, scaler_target, scaler_cov, sent_idx_full, adx_idx_full, 
     v_overlap, te_overlap) = load_and_split_data(
         target_col='Close', 
         max_lookback=max_lookback_needed
     )

    for scenario in scenarios:
        ColorLog.extreme(f"SCENARIO STARTED: {scenario.upper()}", color="#FF1493", border_color="#FF1493")
        
        (train_data, train_regs, val_data, val_regs, test_data, test_regs,
         input_dim, use_weighted_loss, current_sent_idx, current_adx_idx, current_scaler_cov) = setup_scenario_context(
            scenario, train_vals_full, train_regs_full, 
            val_vals_full, val_regs_full, 
            test_vals_full, test_regs_full, 
            input_dim_full, sent_idx_full, adx_idx_full, scaler_cov
        )

        for pred_len in prediction_horizons:
            ColorLog.extreme(f"HORIZON STARTED: {pred_len} DAYS", color="#00BFFF", border_color="#00BFFF")
            
            # 🎨 Base Comparison
            evaluate_naive_baseline(scenario, pred_len, test_data, scaler_target, te_overlap, OUTPUT_FILE)

            for lookback in lookback_windows:
                ColorLog.extreme(f"LOOKBACK WINDOW STARTED: {lookback} DAYS", color="#32CD32", border_color="#32CD32")
                
                # Dynamic column adjustment based on Scenario
                current_feature_cols = raw_cols if scenario == "With Sentiment" else [c for c in raw_cols if c != 'Sentiment_Score']
                
                # ⚙️ Independent Tuning, Uncertainty Estimation, and Phase Aggregation
                window_results = run_ablation_retraining(scenario, lookback, pred_len, train_data, train_regs, val_data, val_regs, v_overlap,
                                                train_marks_full, val_marks_full, test_data, test_marks_full, test_regs,
                                                input_dim, current_sent_idx, current_adx_idx, 
                                                use_weighted_loss, scaler_target, te_overlap, current_feature_cols, current_scaler_cov) # 🚀 Pass scaler_cov
                
                all_metrics.extend(window_results)
                
                # 8️⃣ Memory Optimization: Clear GPU cache and garbage collect after each window
                torch.cuda.empty_cache()
                gc.collect()

    # ⚙️ Final Statistical Reporting (Updated for a single static split)
    if all_metrics:
        ColorLog.extreme("ALL TASKS COMPLETED - GENERATING LEADERBOARD", color="#FF4500", border_color="#FF4500")
        
        final_df = pd.DataFrame(all_metrics)
        
        ColorLog.section("📊 FINAL LEADERBOARD")
        for _, row in final_df.iterrows():
            scenario_str = f"{ColorLog.OKCYAN}{row['Scenario']:<18}{ColorLog.ENDC}"
            model_str = f"{ColorLog.BOLD}{row['Model']:<25}{ColorLog.ENDC}"
            horizon_str = f"H: {row['Horizon']:<2} | L: {row['Lookback']:<3}"
            # ⚙️ MINIMAL CHANGE: Updated formatting from .2f to .6f
            mse_str = f"Scaled MSE: {ColorLog.OKGREEN}{row['Scaled MSE (%)']:.6f}%{ColorLog.ENDC}"
            cprint(f"{scenario_str} &nbsp;|&nbsp; {model_str} &nbsp;|&nbsp; {horizon_str} &nbsp;|&nbsp; {mse_str}")
        cprint(f"{ColorLog.BOLD}{ColorLog.HEADER}{'='*60}{ColorLog.ENDC}")

if __name__ == "__main__":
    main()