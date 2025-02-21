import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def prepare_data(csv_file, target_column='Churn', test_size=0.3, random_state=42):
    """Charge et prétraite les données."""
    df = pd.read_csv(csv_file)

    # Identifier les colonnes catégoriques et les encoder
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Encoder la colonne cible si elle est catégorique
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype('category').cat.codes
    
    # Appliquer One-Hot Encoding sur les autres colonnes catégoriques
    df = pd.get_dummies(df, columns=[col for col in categorical_columns if col != target_column], drop_first=True)

    # Séparer X et y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, max_depth=5, random_state=42):
    """Entraîne un modèle Decision Tree."""
    model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et affiche les métriques."""
    y_pred = model.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def save_model(model, filename="decision_tree_model.pkl"):
    """Sauvegarde le modèle entraîné."""
    joblib.dump(model, filename)

def load_model(filename="decision_tree_model.pkl"):
    """Charge un modèle sauvegardé."""
    return joblib.load(filename)
