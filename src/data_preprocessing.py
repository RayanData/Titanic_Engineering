import pandas as pd
import os

def load_data(file_path):
    """
    Charge un fichier CSV dans un DataFrame pandas.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Nettoie les données et prépare les features.
    """
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    
    # Copie et sélection
    df_clean = df[features].copy()
    
    # Encodage (Sex -> 0/1)
    df_clean = pd.get_dummies(df_clean)
    
    # Imputation des valeurs manquantes
    df_clean = df_clean.fillna(0)
    
    return df_clean
