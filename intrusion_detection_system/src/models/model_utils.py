import joblib
from sklearn.metrics import classification_report, confusion_matrix

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    return joblib.load(filepath)

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    print(f"\n=== Evaluation Report for {model_name} ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))