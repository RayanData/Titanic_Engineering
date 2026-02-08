import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import preprocess_data

def test_preprocess_data_columns():
    """
    Vérifie que le nettoyage ne garde QUE les colonnes demandées.
    """
    print("Test en cours : Vérification des colonnes...")
    
    # 1. Création de fausses données
    fake_data = pd.DataFrame({
        "PassengerId": [1, 2],
        "Survived": [0, 1],
        "Pclass": [3, 1],
        "Name": ["Mr. A", "Mrs. B"],
        "Sex": ["male", "female"],
        "Age": [22, 38],
        "SibSp": [1, 1],
        "Parch": [0, 0],
        "Ticket": ["A/5", "PC"],
        "Fare": [7.25, 71.28],
        "Cabin": [None, "C85"],
        "Embarked": ["S", "C"]
    })

    # 2. Exécution de la fonction à tester
    clean_data = preprocess_data(fake_data)

    # 3. Vérifications (Assertions)
    if clean_data.isna().sum().sum() != 0:
        raise AssertionError("Le nettoyage a laissé des valeurs vides !")
    
    
    colonnes_attendues = ["Pclass", "SibSp", "Parch"]
    for col in colonnes_attendues:
        if col not in clean_data.columns:
            raise AssertionError(f"La colonne {col} a disparu !")
            
    if "Sex_male" not in clean_data.columns and "Sex_female" not in clean_data.columns:
         raise AssertionError("L'encodage du sexe (One-Hot) n'a pas fonctionné.")

    print("✅ SUCCÈS : Le nettoyage des données est validé.")

if __name__ == "__main__":
    try:
        test_preprocess_data_columns()
        print("\n--- RÉSULTAT FINAL : TOUS LES TESTS SONT VERTS ---")
    except Exception as e:
        print(f"\n❌ ÉCHEC : {e}")
        sys.exit(1)
