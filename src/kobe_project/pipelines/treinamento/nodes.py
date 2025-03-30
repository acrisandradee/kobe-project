import pandas as pd
from pycaret.classification import setup, create_model, predict_model
from sklearn.metrics import log_loss, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from typing import Any
from sklearn.metrics import confusion_matrix, f1_score, log_loss, classification_report



def treinar_logistica(base_train: pd.DataFrame, base_test: pd.DataFrame) -> Any:
    mlflow.set_experiment("Treinamento")
    with mlflow.start_run(run_name="Regressao_Logistica"):
        s = setup(data=base_train, target='shot_made_flag', session_id=123, verbose=False)
        model = create_model("lr")
        predictions = predict_model(model, data=base_test)

        y_true = base_test["shot_made_flag"]
        y_pred_proba = predictions.get("prediction_score", predictions.get("Score"))

        logloss = log_loss(y_true, y_pred_proba)
        mlflow.log_metric("log_loss_logistica", logloss)

        return model


def treinar_arvore(base_train: pd.DataFrame, base_test: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_experiment("Treinamento")
    with mlflow.start_run(run_name="Arvore_Decisao"):
        s = setup(data=base_train, target='shot_made_flag', session_id=123, verbose=False)
        model = create_model("dt")
        predictions = predict_model(model, data=base_test)

        # Log de métricas
        mlflow.log_metric("log_loss", log_loss(base_test["shot_made_flag"], predictions["prediction_score"]))
        mlflow.log_metric("f1_score", f1_score(base_test["shot_made_flag"], predictions["prediction_label"]))

        # Salva o modelo treinado
        mlflow.sklearn.log_model(model, "model")

        return predictions  

import tempfile
import os

def avaliar_modelo(predictions):
    cm = confusion_matrix(predictions['shot_made_flag'], predictions['prediction_label'])
    print("Matriz de Confusão:")
    print(cm)

    print("\nRelatório de Classificação:")
    print(classification_report(predictions['shot_made_flag'], predictions['prediction_label']))

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")

    # Salvar figura temporária e logar no MLflow
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_path = os.path.join(tmp_dir, "confusion_matrix.png")
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path, artifact_path="plots")

    plt.show()

