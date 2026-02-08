import os
import sys
import pandas as pd

from data_preprocessing import load_data, preprocess_data
from model_training import train_model, evaluate_model, save_model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    root_dir = os.path.dirname(current_dir)
    

    TRAIN_PATH = os.path.join(root_dir, 'data', 'train.csv')
    MODEL_PATH = os.path.join(root_dir, 'data', 'model.pkl')

    print(f"Working directory: {root_dir}")
    print(f"Looking for data at: {TRAIN_PATH}")

    try:
        # 1. Chargement
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"ERREUR : Le fichier {TRAIN_PATH} est introuvable. Veuillez créer un dossier 'data' à la racine du projet et y mettre train.csv.")
        
        print(f"Loading data...")
        df_train = load_data(TRAIN_PATH)

        # 2. Prétraitement
        print("Preprocessing data...")
        X = preprocess_data(df_train)
        y = df_train["Survived"]

        # 3. Entraînement
        print("Training model...")
        model = train_model(X, y)

        # 4. Évaluation
        accuracy = evaluate_model(model, X, y)
        print(f"Model accuracy: {accuracy:.4f}")

        # 5. Sauvegarde
        save_model(model, MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
