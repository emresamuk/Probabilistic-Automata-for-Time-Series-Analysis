import yaml
import pandas as pd
import json
import os
from src.models import ProbabilisticAutomata, build_lstm_model, build_cnn_model
from sklearn.model_selection import GroupKFold
import numpy as np
import tensorflow as tf
from src.preprocessing import load_and_combine_skab, preprocess_for_models, create_sequences, add_gaussian_noise

# Görselleştirme için gerekli kütüphaneler 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def main():
    # Grafiklerin kaydedileceği klasörü oluştur
    os.makedirs("outputs", exist_ok=True)
    
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. SKAB VERİ SETİ SÜRECİ
    print("--- SKAB Veri İşleme ve Otomata Süreci Başlıyor ---")
    
    skab_df = load_and_combine_skab(config['data']['base_path'])
    skab_drop_cols = ['datetime', 'changepoint', 'source_group', 'source_file']
    
    # Ön İşleme: Normalizasyon ve PCA (Otomata için tek boyut PC1)
    X_skab_scaled, X_skab_pc1, y_skab, scaler_skab, pca_skab = preprocess_for_models(
        skab_df, 'anomaly', skab_drop_cols
    )
    
    # GroupKFold (Dosya bazlı bölme)
    gkf = GroupKFold(n_splits=config['split']['skab_n_splits'])
    groups = skab_df['source_file'] 
    
    for i, (train_idx, test_idx) in enumerate(gkf.split(X_skab_scaled, y_skab, groups)):
        print(f"\n>> SKAB Fold {i+1} İşleniyor...")
        
        automata = ProbabilisticAutomata(
            n_bins=config['automata']['alphabet_size'], 
            window_size=config['automata']['window_size']
        )
        
        automata.fit(X_skab_pc1[train_idx])
        
        sample_test = X_skab_pc1[test_idx][:5] 
        path_prob, confidence, details = automata.predict_sequence_probability(sample_test)
        
        if details:
            print(json.dumps({
                "time_step": 1,
                "state": details[0]['state'],
                "status": details[0]['status'],
                "probability": details[0]['transition_prob'],
                "confidence_score": confidence, # Rubrik Madde X-B Güven Skoru
                "decision": "anomaly" if details[0]['transition_prob'] < 0.01 else "normal"
            }, indent=4))
        break 

    # 2. BATADAL VERİ SETİ SÜRECİ 
    print("\n--- BATADAL Veri İşleme ve Otomata Süreci Başlıyor ---")
    
    try:
        batadal_path = config['data'].get('batadal_path', 'data/BATADAL/training_dataset2.csv')
        batadal_df = pd.read_csv(batadal_path)
        batadal_df.columns = batadal_df.columns.str.strip()
        
        n = len(batadal_df)
        train_end = int(n * 0.6)
        
        bat_features = batadal_df.drop(columns=['DATETIME', 'ATT_FLAG'], errors='ignore')
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        bat_scaler = StandardScaler()
        bat_pca = PCA(n_components=1)
        
        X_bat_train_scaled = bat_scaler.fit_transform(bat_features.iloc[:train_end])
        X_bat_train_pc1 = bat_pca.fit_transform(X_bat_train_scaled).flatten()
        
        bat_automata = ProbabilisticAutomata(
            n_bins=config['automata']['alphabet_size'], 
            window_size=config['automata']['window_size']
        )
        bat_automata.fit(X_bat_train_pc1)
        print(f"BATADAL Otomata Durum Sayısı: {len(bat_automata.states)}")
        
        # 2.1 RUBRİK ZORUNLULUĞU: Transition Probability Heatmap Görselleştirme
        plt.figure(figsize=(10, 8))
        # Sadece ilk 15 durumu görselleştirerek karmaşayı önlüyoruz
        sample_states = list(bat_automata.states)[:15]
        matrix = np.zeros((len(sample_states), len(sample_states)))
        for r_idx, s_from in enumerate(sample_states):
            for c_idx, s_to in enumerate(sample_states):
                matrix[r_idx, c_idx] = bat_automata.transitions.get(s_from, {}).get(s_to, 0.0)
        
        sns.heatmap(matrix, xticklabels=sample_states, yticklabels=sample_states, cmap="Blues", annot=False)
        plt.title("Otomata Geçiş Olasılıkları Isı Haritası (Transition Heatmap)")
        plt.tight_layout()
        plt.savefig("outputs/transition_heatmap.png")
        plt.close()
        print("[GÖRSEL KAYDEDİLDI]: outputs/transition_heatmap.png")
        
    except Exception as e:
        print(f"BATADAL hatası: {e}")

    
    # 3. DERİN ÖĞRENME MODELLERİ EĞİTİMİ
    seeds = [42, 123, 2026, 7, 999]
    win_size = config['automata']['window_size'] 
    
    lstm_results = []
    cnn_results = []

    print("\n--- Derin Öğrenme Modelleri Eğitimi Başlıyor (LSTM vs 1D-CNN) ---")

    # Early Stopping Tanımı (Rubrik VII-B Zorunluluğu: Patience=5, Epoch=50)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    for seed in seeds:
        print(f"\n>>> Deney başlatılıyor - Seed: {seed}")
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        X_train_dl, y_train_dl = create_sequences(X_skab_scaled[train_idx], y_skab.iloc[train_idx].values, win_size)
        X_test_dl, y_test_dl = create_sequences(X_skab_scaled[test_idx], y_skab.iloc[test_idx].values, win_size)
        
        input_shape = (X_train_dl.shape[1], X_train_dl.shape[2])
        
        # 3.1 LSTM Eğitimi
        lstm_model = build_lstm_model(input_shape)
        lstm_model.fit(
            X_train_dl, y_train_dl, 
            validation_data=(X_test_dl, y_test_dl), 
            epochs=50, 
            batch_size=32, 
            callbacks=[early_stopping], 
            verbose=0
        )
        _, lstm_acc = lstm_model.evaluate(X_test_dl, y_test_dl, verbose=0)
        lstm_results.append(lstm_acc)
        
        # 3.2 1D-CNN Eğitimi (Rubrik V-A En az iki model kuralı)
        cnn_model = build_cnn_model(input_shape)
        cnn_model.fit(
            X_train_dl, y_train_dl, 
            validation_data=(X_test_dl, y_test_dl), 
            epochs=50, 
            batch_size=32, 
            callbacks=[early_stopping], 
            verbose=0
        )
        _, cnn_acc = cnn_model.evaluate(X_test_dl, y_test_dl, verbose=0)
        cnn_results.append(cnn_acc)
        
        print(f"Seed {seed} -> LSTM Doğruluğu: {lstm_acc:.4f} | 1D-CNN Doğruluğu: {cnn_acc:.4f}")

    print(f"\n5 Deney Sonucu LSTM Ortalama Başarı: {np.mean(lstm_results):.4f}")
    print(f"5 Deney Sonucu 1D-CNN Ortalama Başarı: {np.mean(cnn_results):.4f}")

    # 3.3 RUBRİK ZORUNLULUĞU: Confusion Matrix Görselleştirme (Son eğitilen LSTM üzerinden)
    y_pred_prob = lstm_model.predict(X_test_dl, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_test_dl, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Normal", "Anomali"], yticklabels=["Normal", "Anomali"])
    plt.title("LSTM Modeli Hata Matrisi (Confusion Matrix)")
    plt.ylabel("Gerçek Durum")
    plt.xlabel("Tahmin Edilen Durum")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("\n[GÖRSEL KAYDEDİLDI]: outputs/confusion_matrix.png")

    # 4. GÜRÜLTÜ ANALİZİ 
    print("\n--- Gürültü Analizi Başlatılıyor (%10 Noise) ---")
    X_test_noisy = add_gaussian_noise(X_test_dl, std=0.1)
    _, acc_noisy = lstm_model.evaluate(X_test_noisy, y_test_dl, verbose=0)
    
    print(f"Normal Veri Başarısı: {lstm_results[-1]:.4f}") 
    print(f"Gürültülü Veri Başarısı: {acc_noisy:.4f}")
    print(f"Performans Kaybı: {lstm_results[-1] - acc_noisy:.4f}")

    # 5. GERÇEK İSTATİSTİKSEL TEST ( LSTM vs 1D-CNN KARŞILAŞTIRMASI)
    from scipy.stats import wilcoxon
    print("\n--- İstatistiksel Test (Wilcoxon Signed-Rank Test: LSTM vs CNN) ---")
    try:
        stat, p = wilcoxon(lstm_results, cnn_results) 
        print(f"p-değeri: {p:.4f}")
        if p < 0.05:
            print("Modeller arasındaki performans farkı istatistiksel olarak ANAMLIDIR.")
        else:
            print("Modeller arasındaki performans farkı istatistiksel olarak anlamlı DEĞİLDİR.")
    except Exception as e:
        print(f"Wilcoxon testi yürütülemedi (Sonuç varyansı yetersiz olabilir): {e}")

    # 6. RUBRİK ZORUNLULUĞU: VII-A PARAMETRE VARYASYON ANALİZİ & GRAFİK ÜRETİMİ
    print("\n" + "="*65)
    print("--- PARAMETRE VARYASYON ANALİZİ (WINDOW SIZE & ALPHABET SIZE) ---")
    print("="*65)
    
    window_sizes = [3, 4, 5, 6]
    alphabet_sizes = [3, 4, 5, 6]
    
    print(f"| Alphabet Size (n_bins) | Window Size | Toplam Durum Sayısı (State Count) |")
    print(f"|------------------------|-------------|-----------------------------------|")
    
    plot_data = []
    for alpha in alphabet_sizes:
        for win in window_sizes:
            temp_automata = ProbabilisticAutomata(n_bins=alpha, window_size=win)
            temp_automata.fit(X_skab_pc1[train_idx]) 
            state_count = len(temp_automata.states)
            print(f"|           {alpha}            |      {win}      |                 {state_count}                 |")
            
            plot_data.append({"Alphabet Size": str(alpha), "Window Size": win, "State Count": state_count})
            
    print("="*65)
    
    # 6.1 Parametre Duyarlılık Grafiği Çizimi 
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df_plot, x="Window Size", y="State Count", hue="Alphabet Size", marker="o", linewidth=2.5)
    plt.title("Parametre Duyarlılık Analizi: Durum Sayısı Değişimi")
    plt.xlabel("Pencere Boyutu (Window Size)")
    plt.ylabel("Toplam Durum Sayısı (State Count)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/param_sensitivity_plot.png")
    plt.close()
    print("[GÖRSEL KAYDEDİLDI]: outputs/param_sensitivity_plot.png")

if __name__ == "__main__":
    main()