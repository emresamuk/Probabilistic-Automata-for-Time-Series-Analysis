import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_combine_skab(base_path):
    all_dfs = []
  
    for group in ['valve1', 'valve2']:
        folder_path = os.path.join(base_path, 'SKAB', group)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file_path in csv_files:
            df = pd.read_csv(file_path, sep=';')
           
            df['source_group'] = group
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)
     
    combined_df = pd.concat(all_dfs, ignore_index=True) 
    combined_df.ffill(inplace=True) 
    return combined_df

def add_gaussian_noise(X, mean=0, std=0.1):
    """Deney senaryosu: Gürültü eklenmiş veri [cite: 282]"""
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def preprocess_for_models(df, target_column, drop_columns, scaler=None, pca=None):
    X = df.drop(columns=[target_column] + drop_columns)
    y = df[target_column]
    
    # Normalizasyon
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
     
    if pca is None:
        pca = PCA(n_components=1)
        X_pc1 = pca.fit_transform(X_scaled)
    else:
        X_pc1 = pca.transform(X_scaled)
    
    return X_scaled, X_pc1, y, scaler, pca

def create_sequences(data, target, window_size):
    X, y = [], []  
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)