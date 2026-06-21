class AccuracyOptimizer:
    def __init__(self, uncertainty_floor=0.1):
        self.floor = uncertainty_floor

    def fuse_and_compare(self, tb_analysis, sgd_prediction):
        """Convex structural information blend minimizing localized noise."""
        p_current = tb_analysis['score']
        s_current = tb_analysis['subjectivity']
        
        current_weight = max(s_current, self.floor)
        trend_weight = 1.0 - current_weight
        
        fused_score = (p_current * current_weight) + (sgd_prediction * trend_weight)
        
        return {
            "final_polarity": round(fused_score, 4),
            "bias_leaning": "Contextual" if current_weight > 0.5 else "Trend-Based",
            "reliability": round(current_weight, 2)
        }