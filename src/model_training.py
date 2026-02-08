import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X, y):
    """
    Configure et entraîne le modèle Random Forest.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """
    Retourne la précision du modèle.
    """
    predictions = model.predict(X)
    return accuracy_score(y, predictions)

def save_model(model, path):
    """
    Sauvegarde le modèle sur le disque.
    """
    joblib.dump(model, path)
