import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def prepare_data(filename):
    data = pd.read_csv(filename)

    # Sélectionner les caractéristiques et la cible
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    # Convertir les variables catégorielles en numériques
    X = pd.get_dummies(X)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # Prédictions
    y_pred = model.predict(X_test)

    # Rapport de classification
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de Confusion:")
    print(cm)

    # Matrice de corrélation
    corr_matrix = X_test.corr()
    print("Matrice de Corrélation:")
    print(corr_matrix)


def save_model(model, filename="model.pkl"):
    joblib.dump(model, filename)


def load_model(filename="model.pkl"):
    return joblib.load(filename)
