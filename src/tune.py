"""Recherche de C : autolog crée un run parent et un run enfant par valeur."""

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Préparer les mêmes données que dans train.py
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
y = 1 - y

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Configurer MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cancer-sein")

# Enregistrement automatique des essais
mlflow.sklearn.autolog(
    max_tuning_runs=None,
    log_models=False
)

# Créer le modèle
modele = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Valeurs de C à tester
grille = {
    "logisticregression__C": [0.01, 0.1, 1.0, 3.0, 10.0]
}

# Validation croisée à 5 plis
recherche = GridSearchCV(
    modele,
    grille,
    cv=5,
    scoring="recall"
)

# Enregistrer la recherche dans MLflow
mlflow.start_run(run_name="recherche-C")

recherche.fit(X_tr, y_tr)

mlflow.end_run()

print(
    f"meilleur C : {recherche.best_params_['logisticregression__C']} | "
    f"recall moyen (validation croisée) : {recherche.best_score_:.4f}"
)