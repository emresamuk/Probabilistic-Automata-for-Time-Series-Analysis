import yaml
import pandas as pd
import json
from src.preprocessing import load_and_combine_skab, preprocess_for_models
from src.models import ProbabilisticAutomata
from sklearn.model_selection import GroupKFold
import numpy as np
import tensorflow as tf
from src.models import build_lstm_model, build_cnn_model
from src.preprocessing import create_sequences

def main():
    # 1. Konfigürasyonu Yükle
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- SKAB VERİ SETİ SÜRECİ ---
    print("--- SKAB Veri İşleme ve Otomata Süreci Başlıyor ---")
    
    # Verileri birleştirme (concat ve ek sütunlar)
    skab_df = load_and_combine_skab(config['data']['base_path'])
    skab_drop_cols = ['datetime', 'changepoint', 'source_group', 'source_file'] # 
    
    # Ön İşleme: Normalizasyon ve PCA (Otomata için tek boyut PC1)
    X_skab_scaled, X_skab_pc1, y_skab, scaler_skab, pca_skab = preprocess_for_models(
        skab_df, 'anomaly', skab_drop_cols
    )
    
    # Bölme Stratejisi: GroupKFold (Dosya bazlı bölme)
    gkf = GroupKFold(n_splits=config['split']['skab_n_splits'])
    groups = skab_df['source_file'] 
    
    for i, (train_idx, test_idx) in enumerate(gkf.split(X_skab_scaled, y_skab, groups)):
        print(f"\n>> SKAB Fold {i+1} İşleniyor...")
        
        # Modeli oluştur (Parametreler: window size=4, alphabet size=3) 
        automata = ProbabilisticAutomata(
            n_bins=config['automata']['alphabet_size'], 
            window_size=config['automata']['window_size']
        )
        
        # Eğitim ve Olasılık Hesaplama
        automata.fit(X_skab_pc1[train_idx])
        
        # Test ve Açıklanabilirlik
        sample_test = X_skab_pc1[test_idx][:5] 
        path_prob, details = automata.predict_sequence_probability(sample_test)
        
        if details:
            print(json.dumps({
                "time_step": 1,
                "state": details[0]['state'],
                "status": details[0]['status'],
                "probability": details[0]['transition_prob'],
                "decision": "anomaly" if details[0]['transition_prob'] < 0.01 else "normal"
            }, indent=4))
        break 

    # --- BATADAL VERİ SETİ SÜRECİ ---
    print("\n--- BATADAL Veri İşleme ve Otomata Süreci Başlıyor ---")
    
    try:
        batadal_df = pd.read_csv(config['data']['batadal_path'])
        batadal_df.columns = batadal_df.columns.str.strip() # Boşlukları temizle
        
        # Zaman sıralı bölme: %60 Train, %20 Val, %20 Test
        n = len(batadal_df)
        train_end = int(n * 0.6)
        val_end = train_end + int(n * 0.2)
        
        X_bat_train_pc1 = X_skab_pc1[:train_end] # Örnekleme
        
        bat_automata = ProbabilisticAutomata(
            n_bins=config['automata']['alphabet_size'], 
            window_size=config['automata']['window_size']
        )
        bat_automata.fit(X_bat_train_pc1)
        print(f"BATADAL Otomata Durum Sayısı: {len(bat_automata.states)}")
        
    except Exception as e:
        print(f"BATADAL hatası: {e}")

   # 1. Ayarlar ve Seed Listesi
    from src.preprocessing import create_sequences # Bunu buraya ekle
    
    seeds = [42, 123, 2026, 7, 999]
    # 'config' artık burada tanınır çünkü main() fonksiyonunun içindeyiz
    win_size = config['automata']['window_size'] 
    all_results = []

    print("\n--- Derin Öğrenme Modelleri Eğitimi Başlıyor ---")

    for seed in seeds:
        print(f"\n>>> Deney başlatılıyor - Seed: {seed}")
        
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Veriyi DL formatına getir
        X_train_dl, y_train_dl = create_sequences(X_skab_scaled[train_idx], y_skab.iloc[train_idx].values, win_size)
        X_test_dl, y_test_dl = create_sequences(X_skab_scaled[test_idx], y_skab.iloc[test_idx].values, win_size)
        
        # 2. LSTM Modelini Kur ve Eğit
        input_shape = (X_train_dl.shape[1], X_train_dl.shape[2])
        model = build_lstm_model(input_shape)
        
        # Epoch sayısını test için 5 yapabilirsin, finalde 10-20 yaparsın
        model.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32, verbose=0)
        
        # 3. Değerlendirme
        loss, acc = model.evaluate(X_test_dl, y_test_dl, verbose=0)
        print(f"Seed {seed} için LSTM Doğruluğu: {acc:.4f}")
        all_results.append(acc)

    if all_results:
        print(f"\n5 Deney Sonucu Ortalama Başarı: {np.mean(all_results):.4f}")

# DİKKAT: main() fonksiyonu burada bitiyor (girinti en sola geliyor)

if __name__ == "__main__":
    main()