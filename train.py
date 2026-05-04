import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    mlflow.set_experiment("Lab_Experiment")
    
    with mlflow.start_run():
        # Load Data
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

        # Hyperparameters
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)

        # Model
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # Metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Save Artifacts (DVC will track the data, MLflow tracks the model)
        mlflow.sklearn.log_model(model, "random-forest-model")
        joblib.dump(model, "model.pkl")
        
        return accuracy

if __name__ == "__main__":
    train_model()