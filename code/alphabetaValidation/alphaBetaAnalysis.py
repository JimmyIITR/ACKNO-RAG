import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join, abspath
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.svm import SVC
import gc

# Set up paths
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import selectData

# Set memory limits
os.environ["OMP_NUM_THREADS"] = "1"  # Reduce parallel memory usage
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def load_and_clean_data():
    """Handle both list and dict data formats"""
    MAX_SAMPLES = 80000
    random.seed(42)
    
    def safe_sample(data, max_size):
        return random.sample(data, min(len(data), max_size))
    
    data_bins = {
        'true_hor': [],
        'false_hor': [],
        'true_ver': [],
        'false_ver': []
    }

    RESULT_PATH = selectData.dataInLogs()
    
    with open(RESULT_PATH, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                data = entry.get('data', {})
                
                # Skip entries without proper data structure
                if not isinstance(data, dict):
                    continue
                
                # Handle horizontal data
                for key in ['true_true_horizontal', 'true_false_horizontal']:
                    values = [x for x in data.get(key, []) if x > 0]
                    target = 'true_hor' if 'true_true' in key else 'false_hor'
                    data_bins[target].extend(values)
                
                # Handle vertical data
                for key in ['true_true_vertical', 'true_false_vertical']:
                    values = [x for x in data.get(key, []) if x > 0]
                    target = 'true_ver' if 'true_true' in key else 'false_ver'
                    data_bins[target].extend(values)
                
                # Early exit check
                if all(len(v) > MAX_SAMPLES for v in data_bins.values()):
                    break
                    
            except Exception as e:
                continue

    # Final sampling and conversion
    return {
        'true_hor': np.array(safe_sample(data_bins['true_hor'], MAX_SAMPLES)),
        'false_hor': np.array(safe_sample(data_bins['false_hor'], MAX_SAMPLES)),
        'true_ver': np.array(safe_sample(data_bins['true_ver'], MAX_SAMPLES)),
        'false_ver': np.array(safe_sample(data_bins['false_ver'], MAX_SAMPLES))
    }

def analyze_with_svm(true_data, false_data, label):
    """Fixed sample size handling for SVM"""
    try:
        # Balance classes
        min_len = min(len(true_data), len(false_data))
        if min_len < 10:
            print(f"Insufficient data for {label}")
            return None
            
        # Create proper 2D input format
        X = np.vstack([  # Changed from hstack to vstack
            true_data[:min_len].reshape(-1, 1),
            false_data[:min_len].reshape(-1, 1)
        ])
        y = np.concatenate([  # Use concatenate instead of hstack
            np.zeros(min_len),
            np.ones(min_len)
        ])
        
        # Verify shapes
        print(f"Debug: X shape {X.shape}, y shape {y.shape}")  # Should show (2*min_len, 1) and (2*min_len,)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        from sklearn.svm import LinearSVC
        model = LinearSVC(class_weight='balanced', max_iter=1000)
        model.fit(X_scaled, y)
        
        # Calculate threshold
        decision_boundary = -model.intercept_[0]/model.coef_[0][0]
        threshold = scaler.inverse_transform([[decision_boundary]])[0][0]
        
        return round(threshold, 2)
        
    except Exception as e:
        print(f"{label} analysis error: {str(e)}")
        return None

def main():
    # Load and clean data
    data = load_and_clean_data()
    
    print("="*50)
    print(f"Data Counts (Max 80k each):")
    print(f"Horizontal - True: {len(data['true_hor']):,}")
    print(f"Horizontal - False: {len(data['false_hor']):,}")
    print(f"Vertical - True: {len(data['true_ver']):,}")
    print(f"Vertical - False: {len(data['false_ver']):,}")
    print("="*50)

    # Horizontal analysis
    print("\nHorizontal Separation:")
    hor_threshold = analyze_with_svm(
        data['true_hor'].flatten(),  # Use flatten() instead of reshape
        data['false_hor'].flatten(),
        'Horizontal'
    )
    print(f"Separation Threshold: {hor_threshold}")

    # Vertical analysis 
    print("\nVertical Separation:")
    ver_threshold = analyze_with_svm(
        data['true_ver'].flatten(),
        data['false_ver'].flatten(),
        'Vertical'
    )
    print(f"Separation Threshold: {ver_threshold}")

    # Clean up
    del data
    gc.collect()

if __name__ == "__main__":
    main()