import json

def format_probabilistic_explanation(time_step, details, confidence_score):
    """
    Rubrik Madde X-F: Zorunlu JSON Çıktı Formatı Oluşturucu.
    Otomata kararlarını insani ve açıklanabilir bir forma dönüştürür.
    """
    if not details:
        return "{}"
        
    first_step = details[0]
    explanation = {
        "time_step": time_step,
        "state": first_step.get("state"),
        "pattern": first_step.get("state"),  # SAX sembolü
        "status": first_step.get("status"),
        "mapped_to": first_step.get("mapped_to"),
        "probability": first_step.get("transition_prob"),
        "confidence_score": confidence_score,
        "decision": "anomaly" if first_step.get("transition_prob") < 0.01 else "normal"
    }
    
    return json.dumps(explanation, indent=4)