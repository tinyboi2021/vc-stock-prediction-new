import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
RESULTS_CSV = "./feature_ablation_results_H16.csv"
OUTPUT_RANKING_CSV = "./model_rankings_final.csv"

def rank_ablation_models(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run the ablation study first.")
        return

    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # Remove duplicates if the study was run multiple times
    df = df.sort_values('Timestamp').drop_duplicates('Scenario', keep='last')

    # 2. Calculate Relative Improvement
    # We compare everything against the '1_Baseline_Only' scenario
    baseline_rmse = df[df['Scenario'] == '1_Baseline_Only']['RMSE'].values[0]
    df['Improvement_vs_Baseline_%'] = ((baseline_rmse - df['RMSE']) / baseline_rmse) * 100

    # 3. Create a Composite Score (Lower is better for RMSE/MAE)
    # We rank primarily on RMSE, then MAE as a tie-breaker
    df['Rank'] = df['RMSE'].rank(ascending=True, method='min').astype(int)
    
    # 4. Sorting
    df = df.sort_values('Rank')

    print("\n" + "="*50)
    print("FEATURE ABLATION RANKING (Horizon: 16D)")
    print("="*50)
    print(df[['Rank', 'Scenario', 'RMSE', 'MAE', 'Improvement_vs_Baseline_%']].to_string(index=False))
    
    # 5. Save Ranking
    df.to_csv(OUTPUT_RANKING_CSV, index=False)
    print(f"\nDetailed rankings saved to: {OUTPUT_RANKING_CSV}")

    # 6. Visualization
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Bar plot for RMSE
    ax = sns.barplot(x='RMSE', y='Scenario', data=df.sort_values('RMSE', ascending=False), palette='viridis')
    plt.title('Informer Model Performance by Feature Scenario (Lower is Better)')
    plt.xlabel('Root Mean Squared Error (RMSE)')
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + (width*0.01), p.get_y() + p.get_height()/2, 
                 f"{df.iloc[::-1]['Improvement_vs_Baseline_%'].values[i]:.2f}% vs Base", 
                 va='center')

    plt.tight_layout()
    plt.savefig('./ablation_ranking_visual.png')
    plt.show()

    return df

if __name__ == "__main__":
    ranked_df = rank_ablation_models(RESULTS_CSV)
    
    # Identify the optimal feature set
    best_scenario = ranked_df.iloc[0]['Scenario']
    print(f"\n🏆 OPTIMAL CONFIGURATION: {best_scenario}")
    print(f"This configuration reduced error by {ranked_df.iloc[0]['Improvement_vs_Baseline_%']:.2f}% over baseline.")