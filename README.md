# projet-mlflow
## Suivi des expériences (MLflow)

Jeu : Breast Cancer Wisconsin (Diagnostic), fourni par scikit-learn.

Expérience MLflow : `cancer-sein` ; modèle enregistré : `cancer-sein`.

- Entraîner : `python src/train.py 3.0` (C en argument)
- Chercher C : `python src/tune.py`
- Interface : `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Registre : `python src/register.py` (alias `champion`)
- Résultats du champion : [reports/champion.md](reports/champion.md)