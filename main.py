import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML avec Decision Tree"
    )
    parser.add_argument(
        "--train", action="store_true", help="Entraîner un modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle"
    )
    parser.add_argument(
        "--save", action="store_true", help="Sauvegarder le modèle"
    )
    parser.add_argument(
        "--load", action="store_true", help="Charger le modèle et prédire"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Préparer les données"
    )

    args = parser.parse_args()

    model = None  # Initialiser le modèle à None

    # Charger le modèle si l'argument --load est passé
    if args.load:
        model = load_model()
        print("✅ Modèle chargé avec succès !")

    if args.prepare:
        # Préparer les données (séparation en train/test)
        X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")
        print("Données préparées avec succès!")

    if args.train:
        # Entraîner le modèle
        X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")
        model = train_model(X_train, y_train)
        print("Modèle entraîné avec succès!")

        if args.save:
            save_model(model)
            print("Modèle sauvegardé!")

    if args.evaluate:
        if model is None:  # Vérifier si le modèle a été chargé ou entraîné
            print("Erreur: Aucun modèle n'est disponible pour l'évaluation.")
        else:
            X_train, X_test, y_train, y_test = prepare_data("churn-bigml-20.csv")
            evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()