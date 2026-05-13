import numpy as np
from pyts.approximation import SymbolicAggregateApproximation
import Levenshtein # pip install python-Levenshtein komutu gerekebilir
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

class ProbabilisticAutomata:
    def __init__(self, n_bins=3, window_size=4):
        self.n_bins = n_bins
        self.window_size = window_size
        self.sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
        self.transitions = {} 
        self.states = set()

    def fit(self, X_pc1):
        X_sax = self.sax.fit_transform(X_pc1.reshape(1, -1))
        sequence = X_sax[0]
        for i in range(len(sequence) - self.window_size):
            state = "".join(sequence[i : i + self.window_size])
            next_state = "".join(sequence[i + 1 : i + 1 + self.window_size])
            if state not in self.transitions: self.transitions[state] = {}
            self.transitions[state][next_state] = self.transitions[state].get(next_state, 0) + 1
            self.states.add(state)
        for state, next_states in self.transitions.items():
            total_output = sum(next_states.values())
            for ns in next_states: next_states[ns] /= total_output
        print(f"Otomata eğitildi. Durum Sayısı: {len(self.states)}")

    def find_nearest_state(self, unseen_state):
        """Unseen Pattern Yönetimi: Levenshtein algoritması [cite: 56]"""
        distances = {state: Levenshtein.distance(unseen_state, state) for state in self.states}
        # En yakın (mesafesi en küçük) durumu bul
        nearest_state = min(distances, key=distances.get)
        return nearest_state, distances[nearest_state]

    def predict_sequence_probability(self, sequence_indices):
        """Path Probability hesaplama [cite: 140, 149]"""
        # Test verisini sembole çevir
        test_sax = self.sax.transform(sequence_indices.reshape(1, -1))[0]
        
        path_prob = 1.0
        details = []

        for i in range(len(test_sax) - self.window_size):
            state = "".join(test_sax[i : i + self.window_size])
            next_state = "".join(test_sax[i + 1 : i + 1 + self.window_size])
            
            status = "seen"
            mapped_to = state
            
            # Eğer durum eğitimde yoksa Levenshtein kullan
            if state not in self.states:
                status = "unseen"
                mapped_to, dist = self.find_nearest_state(state)
            
            # Geçiş olasılığını al, yoksa çok küçük bir değer ver 
            prob = self.transitions.get(mapped_to, {}).get(next_state, 0.0001)
            path_prob *= prob
            
            details.append({
                "state": state,
                "mapped_to": mapped_to,
                "status": status,
                "transition_prob": prob
            })
            
        return path_prob, details

def build_lstm_model(input_shape):
        model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # Anomali mi değil mi? (0-1)
            ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

def build_cnn_model(input_shape):
        model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
            ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model