import pandas as pd
import numpy as np
import os
import logging
import yaml
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_llama_model(repo_id: str, filename: str) -> Llama:
    """Downloads and initializes the Llama 3 GGUF model."""
    logging.info(f"Downloading/Loading GGUF Model from {repo_id}...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    logging.info("Initializing model on GPU...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=2048,      
        verbose=False    
    )
    return llm

def get_sentiment(headline: str, llm: Llama, cache: dict) -> int:
    """Feeds a single headline to LLaMA 3 and extracts the polarity."""
    if headline in cache:
        return cache[headline]
    
    # Truncate to save context window and speed up inference
    safe_headline = " ".join(str(headline).split()[:300])
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a financial analyst. Classify the sentiment of this news headline.
    Return ONLY one word: 'positive', 'negative', or 'neutral'.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Headline: "{safe_headline}"<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    Sentiment:"""
    
    try:
        output = llm(
            prompt,
            max_tokens=5,
            temperature=0.0, 
            stop=["<|eot_id|>"], 
            echo=False
        )
        
        response = output['choices'][0]['text'].strip().lower()
        
        if 'positive' in response: 
            score = 1
        elif 'negative' in response: 
            score = -1
        else: 
            score = 0
            
        cache[headline] = score
        return score
        
    except ValueError:
        return 0

def process_feature_engineering(input_path: str, output_path: str, repo_id: str, filename: str):
    """Orchestrates the sentiment scoring and saves the final dataset."""
    llm = load_llama_model(repo_id, filename)
    sentiment_cache = {}
    
    logging.info(f"Loading cleaned dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    avg_scores = []
    
    logging.info("Analyzing sentiment row by row...")
    pbar = tqdm(df.iterrows(), total=len(df), desc="Rows Processed")
    
    for index, row in pbar:
        text = row.get('news_headlines', '')
        
        if pd.isna(text) or not str(text).strip():
            avg_scores.append(0)
            continue
            
        headlines = [h.strip() for h in str(text).split('|') if h.strip()]
        total_news_in_row = len(headlines)
        daily_polarities = []
        
        for i, h in enumerate(headlines):
            pbar.set_postfix({"Row": index, "News": f"{i+1}/{total_news_in_row}"})
            score = get_sentiment(h, llm, sentiment_cache)
            daily_polarities.append(score)
            
        if not daily_polarities:
            avg_scores.append(0)
        else:
            avg_scores.append(np.mean(daily_polarities))

    # Compile the final dataset
    logging.info("Appending sentiment scores and dropping raw text...")
    df_final = df.drop(columns=['news_headlines'], errors='ignore')
    df_final['Sentiment_Score'] = avg_scores
    
    # Save the output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    logging.info(f"✅ Pipeline execution complete! Final dataset saved to: {output_path}")

if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["feature_engineering"]
            
        process_feature_engineering(
            params["input_path"],
            params["output_path"],
            params["model_repo"],
            params["model_filename"]
        )
        
    except FileNotFoundError:
        logging.error("params.yaml not found.")
        exit(1)
    except KeyError as e:
        logging.error(f"Missing key in params.yaml: {e}")
        exit(1)