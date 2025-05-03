import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join, abspath
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.svm import SVC ,LinearSVC
import gc

# Set up paths
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import selectData

def load_and_clean_data():
    """Load data without sampling limitations"""
    def safe_convert(data):
        return np.array([x for x in data if x > 0])
    
    data_bins = {
        'true_hor': [],
        'false_hor': [],
        'true_ver': [],
        'false_ver': []
    }

    RESULT_PATH = selectData.dataInLogsMain()
    
    with open(RESULT_PATH, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                data = entry.get('data', {})
                
                if not isinstance(data, dict):
                    continue
                
                # Horizontal data
                data_bins['true_hor'].extend(data.get('true_true_horizontal', []))
                data_bins['false_hor'].extend(data.get('true_false_horizontal', []))
                
                # Vertical data
                data_bins['true_ver'].extend(data.get('true_true_vertical', []))
                data_bins['false_ver'].extend(data.get('true_false_vertical', []))
                    
            except Exception as e:
                continue

    # Convert to arrays
    return {
        'true_hor': safe_convert(data_bins['true_hor']),
        'false_hor': safe_convert(data_bins['false_hor']),
        'true_ver': safe_convert(data_bins['true_ver']),
        'false_ver': safe_convert(data_bins['false_ver'])
    }

def plot_separation(true_data, false_data, threshold, label):
    """Visualize the separation with threshold line"""
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    sns.histplot(true_data, color='blue', label='True', kde=True, alpha=0.5)
    sns.histplot(false_data, color='red', label='False', kde=True, alpha=0.5)
    
    # Add threshold line
    plt.axvline(x=threshold, color='green', linestyle='--', 
                label=f'Threshold: {threshold:.2f}')
    
    plt.title(f'{label} Separation Analysis')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{label.lower()}_separation.png')
    plt.close()

def plot_roc_curve(y_true, y_scores, label):
    """Generate ROC curve with AUC score"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{label} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{label.lower()}_roc_curve.png')
    plt.close()

def analyze_with_svm(true_data, false_data, label):
    """LinearSVC version with M1-compatible visualization"""
    try:
        # Balance classes
        min_len = min(len(true_data), len(false_data))
        if min_len < 10:
            print(f"Insufficient data for {label}")
            return None
            
        # Create dataset
        X = np.concatenate([true_data[:min_len], false_data[:min_len]]).reshape(-1, 1)
        y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train LINEAR model
        model = LinearSVC(class_weight='balanced', max_iter=10000)
        model.fit(X_scaled, y)
        
        # Get decision boundary (works for linear models)
        decision_boundary = -model.intercept_[0]/model.coef_[0][0]
        threshold = scaler.inverse_transform([[decision_boundary]])[0][0]
        
        # Generate visualization with density plot
        plt.figure(figsize=(10, 6), dpi=100)
        
        sns.kdeplot(true_data, color='blue', label='True', fill=True)
        sns.kdeplot(false_data, color='red', label='False', fill=True)
        plt.axvline(threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold: {threshold:.2f}')
        
        plt.title(f'{label} Separation (Linear SVM)')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'{label}_separation.png', bbox_inches='tight')
        plt.close()
        
        return round(threshold, 2)
        
    except Exception as e:
        print(f"{label} analysis error: {str(e)}")
        return None

def main():
    data = load_and_clean_data()
    
    print("="*50)
    print("Data Counts:")
    print(f"Horizontal - True: {len(data['true_hor']):,}")
    print(f"Horizontal - False: {len(data['false_hor']):,}")
    print(f"Vertical - True: {len(data['true_ver']):,}")
    print(f"Vertical - False: {len(data['false_ver']):,}")
    print("="*50)

    print("\nHorizontal Separation:")
    hor_threshold = analyze_with_svm(data['true_hor'], data['false_hor'], 'Horizontal')
    print(f"Separation Threshold: {hor_threshold}")

    print("\nVertical Separation:")
    ver_threshold = analyze_with_svm(data['true_ver'], data['false_ver'], 'Vertical')
    print(f"Separation Threshold: {ver_threshold}")

    del data
    gc.collect()

if __name__ == "__main__":
    main()