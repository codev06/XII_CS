import math
import hashlib
import csv
import os
from textblob import TextBlob

class TextBlobPredictor:
    def __init__(self, csv_path="polarity_thresholds.csv", sensitivity=0.15):
        self.sensitivity = sensitivity
        self.csv_path = csv_path
        self.SUB_FACTUAL = 0.30
        
        # Hardcoded anchors protecting heavy emotional state terms from dilution
        self.POWER_ANCHORS = {
            'shattered': (-0.8, 0.9), 'void': (-0.6, 0.7), 'raw': (-0.4, 0.8),
            'betrayed': (-0.9, 1.0), 'stinging': (-0.5, 0.6), 'hollow': (-0.3, 0.7),
            'fuming': (-0.7, 0.9), 'aching': (-0.6, 0.8), 'glow': (0.5, 0.6),
            'dead': (-0.9, 0.8), 'scream': (-0.7, 0.9)
        }
        self.EMOTION_MAP = self._load_thresholds()

    def _load_thresholds(self):
        thresholds = {}
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Missing configuration file: {self.csv_path}")
        with open(self.csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                thresholds[row['emotion']] = (float(row['p_coord']), float(row['s_coord']))
        return thresholds

    def _is_outlier(self, word, avg_len):
        """Filters neutral text fluff while preserving specific emotional keys."""
        w_low = word.lower().strip(".,!?")
        if w_low in self.POWER_ANCHORS: 
            return False
        
        b = TextBlob(word)
        p, s = b.sentiment.polarity, b.sentiment.subjectivity
        
        if abs(p) > 0.05: 
            return False 
        if s < self.SUB_FACTUAL and p == 0.0: 
            return True
        
        return abs(len(word) - avg_len) <= self.sensitivity * avg_len

    def _get_label_match(self, p1, s1):
        """Calculates closest Euclidean target and resolves tie breaks deterministically."""
        if p1 == 0 and s1 == 0: 
            return "Apathy/Flat"

        candidates = []
        for label, (p2, s2) in self.EMOTION_MAP.items():
            dist = math.sqrt((p2 - p1)**2 + (s2 - s1)**2)
            intensity = math.sqrt(p2**2 + s2**2)
            tie_hash = int(hashlib.md5(label.encode()).hexdigest(), 16)
            candidates.append((dist, -s2, -intensity, tie_hash, label))
        
        return min(candidates)[4]

    def analyze(self, text):
        if not text: 
            return None

        words = text.split()
        if not words: 
            return {"score": 0.0, "subjectivity": 0.0, "label": "Apathy/Flat"}
        
        avg_len = sum(len(w) for w in words) / len(words)
        p_acc, s_acc = [], []

        for w in words:
            w_clean = w.lower().strip(".,!?")
            if w_clean in self.POWER_ANCHORS:
                p, s = self.POWER_ANCHORS[w_clean]
                p_acc.append(p); s_acc.append(s)
            elif not self._is_outlier(w, avg_len):
                b = TextBlob(w)
                p_acc.append(b.sentiment.polarity)
                s_acc.append(b.sentiment.subjectivity)

        if not p_acc:
            return {"score": 0.0, "subjectivity": 0.0, "label": "Apathy/Flat"}

        final_p = sum(p_acc) / len(p_acc)
        final_s = sum(s_acc) / len(s_acc)
        
        return {
            "score": round(final_p, 4),
            "subjectivity": round(final_s, 4),
            "label": self._get_label_match(final_p, final_s)
        }