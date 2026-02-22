import pandas as pd
import numpy as np

def analyze_labels():
    print("Loading fraud data and call dates...")
    df_calls = pd.read_csv('data/processed/dataset_labels.csv')
    df_fraud = pd.read_csv('data/final_fraud_data.csv')
    
    df_fraud['date'] = pd.to_datetime(df_fraud['date'])
    df_calls['call_date'] = pd.to_datetime(df_calls['call_date'])
    
    counts = {1: 0, 2: 0, 4: 0, 8: 0, 12: 0, 16: 0, 'any': 0}
    DAYS_PER_QUARTER = 91.25
    
    for idx, call in df_calls.iterrows():
        t_call = call['call_date']
        ticker = call['ticker']
        
        future_frauds = df_fraud[(df_fraud['ticker'] == ticker) & (df_fraud['date'] > t_call)]
        
        if not future_frauds.empty:
            counts['any'] += 1
            t_star = future_frauds['date'].min()
            delta_d = (t_star - t_call).days
            q = int(np.ceil(delta_d / DAYS_PER_QUARTER))
            
            for k in [1, 2, 4, 8, 12, 16]:
                if q <= k:
                    counts[k] += 1
                    
    print("\nPositives per horizon:")
    for k, v in counts.items():
        print(f"K={k} <= {v} positives")

if __name__ == "__main__":
    analyze_labels()
