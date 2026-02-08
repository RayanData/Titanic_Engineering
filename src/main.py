import os
import sys
import pandas as pd

# Import des modules locaux
from data_preprocessing import load_data, preprocess_data
from model_training import train_model, evaluate_model, save_model

def main():
    # Configuration des chemins (Adapté à votre Mac)
    BASE_DIR = os.path.expanduser("~/Desktop")
    TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

    print("--- Démarrage du pipeline ---")

    try:
        # 1. Chargement
        print(f"Chargement de {TRAIN_PATH}...")
        df_train = load_data(TRAIN_PATH)

        # 2. Prétraitement
        print("Nettoyage des données...")
        X = preprocess_data(df_train)
        y = df_train["Survived"]

        # 3. Entraînement
        print("Entraînement du modèle...")
        model = train_model(X, y)

        # 4. Évaluation
        accuracy = evaluate_model(model, X, y)
        print(f"Précision obtenue : {accuracy:.4f}")

        # 5. Sauvegarde
        save_model(model, MODEL_PATH)
        print(f"Modèle sauvegardé sous : {MODEL_PATH}")

    except Exception as e:
        print(f"Erreur critique : {e}")

if __name__ == "__main__":
    main()
