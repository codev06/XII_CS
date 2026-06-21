class ModuleAlphaSGD:
    def __init__(self, learning_rate=0.001):
        self.w = 0.0  
        self.b = 0.0  
        self.lr = learning_rate
        self.iteration = 0

    def update_and_predict(self, polarity_value):
        """Executes stochastic updates and yields the prediction step target (x + 1)."""
        self.iteration += 1
        x = self.iteration
        y_actual = float(polarity_value)
        
        y_pred = self.w * x + self.b
        error = y_pred - y_actual
        
        # Calculate gradients and step parameters downward
        self.w -= self.lr * (2 * error * x)
        self.b -= self.lr * (2 * error)
        
        return self.w * (x + 1) + self.b