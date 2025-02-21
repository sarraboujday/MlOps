import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML avec Decision Tree")
    parser.add_argument("--train", action="store_true", help="Entraîner un modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save", action="store_true", help="Sauvegarder le modèle")
    parser.add_argument("--load", action="store_true", help="Charger le modèle et prédire")

    args = parser.parse_args()

    # Charger les données
    X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")

    if args.train:
        model = train_model(X_train, y_train)
        print(" Modèle entraîné avec succès !")

        if args.evaluate:
            evaluate_model(model, X_test, y_test)

        if args.save:
            save_model(model)
            print(" Modèle sauvegardé !")

    if args.load:
        model = load_model()
        print("✅ Modèle chargé avec succès !")

if __name__ == "__main__":
    main()
