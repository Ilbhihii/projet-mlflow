"""Registre : le meilleur run devient le modèle champion."""

import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer

# 1. Connexion à la base MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Rechercher les meilleurs runs d'entraînement
runs = mlflow.search_runs(
    experiment_names=["cancer-sein"],
    filter_string="attributes.run_name LIKE 'logreg-%'",
    order_by=[
        "metrics.recall DESC",
        "metrics.accuracy DESC"
    ]
)

# 3. Sélectionner le meilleur run
best = runs.iloc[0]

# 4. Récupérer le modèle enregistré dans ce run
model_id = mlflow.get_run(
    best.run_id
).outputs.model_outputs[0].model_id

# 5. Enregistrer le modèle dans le registre
v = mlflow.register_model(
    f"models:/{model_id}",
    "cancer-sein"
)

# 6. Attribuer l'alias champion
MlflowClient().set_registered_model_alias(
    "cancer-sein",
    "champion",
    v.version
)

print(
    f"champion = version {v.version} : "
    f"{best['tags.mlflow.runName']}, "
    f"recall {best['metrics.recall']}, "
    f"accuracy {best['metrics.accuracy']}"
)

# 7. Recharger le modèle champion
champion = mlflow.sklearn.load_model(
    "models:/cancer-sein@champion"
)

# 8. Tester le champion sur 8 observations
X, y = load_breast_cancer(
    return_X_y=True,
    as_frame=True
)

patientes = X.sample(8, random_state=0)

print("prédit :", champion.predict(patientes).tolist())
print("réel :", (1 - y[patientes.index]).tolist())