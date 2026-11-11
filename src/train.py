"""Entraîne une régression logistique et enregistre le run dans MLflow."""

# Bloc 1 : importer les bibliothèques
import sys
import matplotlib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Bloc 2 : préparer les données
matplotlib.use("Agg")

C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
y = 1 - y

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Bloc 3 : entraîner et mesurer
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=C, max_iter=1000)
)

model.fit(X_tr, y_tr)
pred = model.predict(X_te)

m = {
    "accuracy": accuracy_score(y_te, pred),
    "f1": f1_score(y_te, pred),
    "recall": recall_score(y_te, pred),
}

m = {k: round(float(v), 4) for k, v in m.items()}
print(m)

# Bloc 4 : ouvrir un run MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cancer-sein")
mlflow.start_run(run_name=f"logreg-C={C}")

# Bloc 5 : enregistrer les résultats et fermer le run
train = mlflow.data.from_pandas(
    X_tr.assign(malin=y_tr),
    targets="malin",
    name="wdbc-train"
)

mlflow.log_input(train, context="training")

mlflow.log_params({
    "C": C,
    "max_iter": 1000,
    "test_size": 0.2
})

mlflow.log_metrics(m)

fig = ConfusionMatrixDisplay.from_predictions(y_te, pred).figure_
mlflow.log_figure(fig, "matrice_confusion.png")

mlflow.sklearn.log_model(
    model,
    name="model",
    input_example=X_te.head(2)
)

mlflow.end_run()