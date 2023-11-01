import numpy as np

class LogisticRegression:
    def __init__(self, 
                learning_rate=0.001, 
                num_epochs=1000, 
                regularization=None, 
                lambda_reg=0.01, 
                gamma=2.0, 
                alpha=None,
                class_weights=None):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None   
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        softmax_vals = exp_z / exp_z.sum(axis=1, keepdims=True)
        return np.clip(softmax_vals, 1e-7, 1 - 1e-7)

    def compute_loss(self, y, probs):
        num_samples = len(y)
        clipped_probs = np.clip(probs, 1e-7, 1 - 1e-7)
        alpha_values_for_samples = np.array(self.alpha)[y]
        
        focal_loss_core = -(alpha_values_for_samples * np.log(clipped_probs[range(num_samples), y])) * (1 - clipped_probs[range(num_samples), y]) ** self.gamma
        loss = np.sum(focal_loss_core * self.class_weights[y])
        
        if self.regularization == "L1":
            reg_loss = self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == "L2":
            reg_loss = 0.5 * self.lambda_reg * np.sum(self.weights**2)
        else:
            reg_loss = 0
        
        return np.mean(loss) + reg_loss

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        if self.class_weights is None:
            self.class_weights = np.ones(num_classes)
        else:
            self.class_weights = np.array(self.class_weights)
            
        if self.alpha is None:
            self.alpha = np.ones(num_classes)

        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        alpha_values_for_samples = np.array(self.alpha)[y]
        
        for epoch in range(self.num_epochs):
            logits = X.dot(self.weights) + self.bias
            probs = self.softmax(logits)
            
            gradient_w = -1/num_samples * X.T.dot(self.class_weights[y].reshape(-1,1) * (alpha_values_for_samples * (1 - probs[range(num_samples), y]) ** self.gamma).reshape(-1,1) * (np.eye(num_classes)[y] - probs))
            gradient_b = -1/num_samples * np.sum(self.class_weights[y].reshape(-1,1) * (alpha_values_for_samples * (1 - probs[range(num_samples), y]) ** self.gamma).reshape(-1,1) * (np.eye(num_classes)[y] - probs), axis=0)

            if self.regularization == "L1":
                gradient_w += self.lambda_reg * np.sign(self.weights)
            elif self.regularization == "L2":
                gradient_w += self.lambda_reg * self.weights

            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

            if epoch % 100 == 0:
                loss = self.compute_loss(y, probs)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        logits = X.dot(self.weights) + self.bias
        probs = self.softmax(logits)
        output = np.argmax(probs, axis=1)
        return output
        
    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "regularization": self.regularization,
            "lambda_reg": self.lambda_reg,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "class_weights": self.class_weights
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
