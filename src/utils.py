import os

def log_experiment_summary(model_name, mean_accuracy, loss=None):
    """
    Rubrik Madde VIII-A: Deney Takibi ve Otomatik Loglama Yardımcısı.
    """
    os.makedirs("outputs", exist_ok=True)
    log_path = "outputs/experiment_logs.txt"
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== Model: {model_name} ===\n")
        f.write(f"Ortalama Başarı (Accuracy): {mean_accuracy:.4f}\n")
        if loss is not None: 
            f.write(f"Son Kayıp (Loss): {loss}\n")
        f.write("-" * 40 + "\n")
    print(f"[LOG KAYDEDİLDİ]: {log_path}")