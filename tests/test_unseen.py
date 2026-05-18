import unittest
import numpy as np
from src.models import ProbabilisticAutomata

class TestUnseenPatternManagement(unittest.TestCase):
    def setUp(self):
        # Sadece test amaçlı, izole bir mini otomata nesnesi ayağa kaldırıyoruz
        self.automata = ProbabilisticAutomata(n_bins=3, window_size=3)
        # Sınıfın states kümesini test senaryosuna göre elle simüle ediyoruz
        self.automata.states = {"aaa", "aab", "abc", "bbb"}
        
    def test_levenshtein_mapping(self):
        # Senaryo: Eğitim sözlüğünde asla bulunmayan "axc" pattern'i geliyor
        unseen_pattern = "axc" 
        
        # models.py içerisindeki Levenshtein mekanizmasını test ediyoruz
        nearest_state, distance = self.automata.find_nearest_state(unseen_pattern)
        
        # Doğrulama (Assertion): "axc"ye en yakın durum "abc" olmalıdır ve düzenleme mesafesi 1 çıkmalıdır
        self.assertEqual(nearest_state, "abc")
        self.assertEqual(distance, 1)
        print(f"\n[UNIT TEST BAŞARILI]: '{unseen_pattern}' girdisi başarıyla en yakın durum olan '{nearest_state}' durumuna eşlendi (Mesafe: {distance})")

if __name__ == '__main__':
    unittest.main()