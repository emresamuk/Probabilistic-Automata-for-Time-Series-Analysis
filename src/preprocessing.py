import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_combine_skab(base_path):
    all_dfs = []
    # valve1 ve valve2 klasörlerini kullan 
    for group in ['valve1', 'valve2']:
        folder_path = os.path.join(base_path, 'SKAB', group)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file_path in csv_files:
            df = pd.read_csv(file_path, sep=';')
            # source_group ve source_file eklenmeli
            df['source_group'] = group
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)
    #concat ıslemi        
    combined_df = pd.concat(all_dfs, ignore_index=True) 
    # Eksik veri kontrolü ve doldurma 
    combined_df.ffill(inplace=True) 
    return combined_df

def add_gaussian_noise(X, mean=0, std=0.1):
    """Deney senaryosu: Gürültü eklenmiş veri [cite: 282]"""
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def preprocess_for_models(df, target_column, drop_columns, scaler=None, pca=None):
    # 1. Gereksiz sütunları at
    X = df.drop(columns=[target_column] + drop_columns)
    y = df[target_column]
    
    # 2. Normalizasyon
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # 3. PCA: Yalnızca train üzerinde fit edilmelidir ve PC1 kullanılmalıdır 
    if pca is None:
        pca = PCA(n_components=1)
        X_pc1 = pca.fit_transform(X_scaled)
    else:
        X_pc1 = pca.transform(X_scaled)
    
    return X_scaled, X_pc1, y, scaler, pca