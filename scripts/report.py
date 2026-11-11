"""Rapport du champion pour GitHub : reports/champion.md et sa figure."""

from pathlib import Path
import mlflow
from mlflow import MlflowClient

# 1. Connexion à la base MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Récupérer la version champion
v = MlflowClient().get_model_version_by_alias(
    "cancer-sein", "champion"
)

# 3. Récupérer le run correspondant
run = mlflow.get_run(v.run_id)

# 4. Créer le dossier reports
Path("reports").mkdir(exist_ok=True)

# 5. Télécharger la matrice de confusion
mlflow.artifacts.download_artifacts(
    run_id=v.run_id,
    artifact_path="matrice_confusion.png",
    dst_path="reports"
)

# 6. Préparer le rapport Markdown
lignes = [
    f"# Modèle champion : cancer-sein, version {v.version}",
    "",
    f"Run `{run.info.run_name}` (C = {run.data.params['C']})",
    "",
    "| Métrique | Valeur |",
    "|---|---|"
]

# Ajouter les métriques
lignes += [
    f"| {k} | {x} |"
    for k, x in sorted(run.data.metrics.items())
]

# Ajouter l'image
lignes += [
    "",
    "![Matrice de confusion](matrice_confusion.png)"
]

# 7. Enregistrer et afficher le rapport
Path("reports/champion.md").write_text(
    "\n".join(lignes) + "\n",
    encoding="utf-8"
)

print(
    Path("reports/champion.md").read_text(
        encoding="utf-8"
    )
)