import numpy as np
import wandb
from sklearn.metrics import f1_score, classification_report

class LogisticRegression:
    def __init__(self, 
                learning_rate=0.001, 
                num_epochs=1000, 
                regularization=None, 
                lambda_reg=0.01, 
                gamma=2.0, 
                alpha=None,
                class_weights=None,
                validation=True,
                logging_notes=" ",
                logging_tag=" ",
                verbose=0):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None   
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.alpha = None
        self.class_weights = class_weights
        self.logging_notes = logging_notes
        self.logging_tag = logging_tag
        self.validation = validation
        self.verbose = verbose

        if verbose:
            # Initialize wandb
            wandb.init(project="ClimateNet",
                    notes=f"{self.logging_notes}",
                    tags=[f"{self.logging_tag}"])
            
            # Log hyperparameters
            wandb.config.learning_rate = learning_rate
            wandb.config.num_epochs = num_epochs
            wandb.config.regularization = regularization
            wandb.config.lambda_reg = lambda_reg
            wandb.config.gamma = gamma
            wandb.config.alpha = alpha
            wandb.config.class_weights = class_weights
            wandb.config.validation = validation

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

    def fit(self, X, y, X_val=None, y_val=None):
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
                train_loss = self.compute_loss(y, probs)
                train_accuracy = np.mean(y == self.predict(X))
                val_accuracy = np.mean(y_val == self.predict(X_val)) if self.validation else None

                if self.verbose:
                    # Log metrics to wandb
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy
                    })
                
                print(f"Epoch {epoch}, Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")
        
        if self.verbose:
            wandb.summary["cross_val_score"] = val_accuracy
            wandb.summary["val_f1"] = f1_score(y_val, self.predict(X_val), average='macro')
        
            report = classification_report(y_val, self.predict(X_val), output_dict=True)
        
            # Convert the classification report into a wandb Table
            table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-score", "Support"])
            for key, value in report.items():
                if key not in ('accuracy', 'macro avg', 'weighted avg'):
                    table.add_data(key, value['precision'], value['recall'], value['f1-score'], value['support'])
        
            # Log the table
            wandb.log({'classification_report': table})

    def get_val_acc(self, X_val, y_val):
        y_pred = self.predict(X_val)
        accuracy = np.mean(y_val == y_pred)
        return accuracy
    
    def predict_proba(self, X):
        logits = X.dot(self.weights) + self.bias
        probs = self.softmax(logits)
        return probs
    
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
