# src/kobe_project/pipelines/treinamento/nodes.py

import mlflow
import mlflow.sklearn
from pycaret.classification import setup, create_model, get_config
from sklearn.metrics import log_loss, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import io

def configurar_pycaret(df_train: pd.DataFrame, target_col: str, random_state: int = 42):
  
    setup(
        data=df_train,
        target=target_col,
        session_id=random_state,
        verbose=False,
        html=False
    )
    # Retorno None proposital: a configuração do PyCaret é global.

def treinar_modelos_e_avaliar(
    df_test: pd.DataFrame
) -> dict:
    """
    Treina 2 modelos (Regressão Logística e Árvore de Decisão) 
    usando PyCaret (já configurado), avalia no df_test e loga no MLflow.
    
    Retorna um dicionário com métricas e arrays para plotar ROC.
    """

    # 1) Criar/treinar modelos
    lr_model = create_model("lr")
    dt_model = create_model("dt")
    
    # 2) Transformar X_test
    X_test = df_test.drop("shot_made_flag", axis=1)
    y_test = df_test["shot_made_flag"]
    pipeline = get_config("pipeline")  # PyCaret pipeline
    X_test_transformed = pipeline.transform(X_test)

    # 3) Avaliar Logística
    y_proba_lr = lr_model.predict_proba(X_test_transformed)
    logloss_lr = log_loss(y_test, y_proba_lr)
    mlflow.log_metric("log_loss_logistica", logloss_lr)
    mlflow.sklearn.log_model(lr_model, artifact_path="modelo_logistico")

    # Prob classe 1
    lr_probs = y_proba_lr[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    auc_lr = roc_auc_score(y_test, lr_probs)

    # 4) Avaliar Árvore
    y_proba_dt = dt_model.predict_proba(X_test_transformed)
    y_pred_dt  = dt_model.predict(X_test_transformed)
    logloss_dt = log_loss(y_test, y_proba_dt)
    f1_dt      = f1_score(y_test, y_pred_dt)

    mlflow.log_metric("log_loss_arvore", logloss_dt)
    mlflow.log_metric("f1_score_arvore", f1_dt)
    mlflow.sklearn.log_model(dt_model, artifact_path="modelo_arvore")

    dt_probs = y_proba_dt[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
    auc_dt = roc_auc_score(y_test, dt_probs)

    # 5) Comparar
    if logloss_lr < logloss_dt:
        vencedor = "Logistica"
    else:
        vencedor = "Arvore"

    mlflow.log_param("modelo_vencedor", vencedor)

    # Retorna dados para plot
    return {
        "lr": {"fpr": fpr_lr, "tpr": tpr_lr, "auc": auc_lr, "logloss": logloss_lr},
        "dt": {"fpr": fpr_dt, "tpr": tpr_dt, "auc": auc_dt, "logloss": logloss_dt, "f1": f1_dt},
    }

def plotar_roc(metrics_dict: dict):
  
    fpr_lr, tpr_lr = metrics_dict["lr"]["fpr"], metrics_dict["lr"]["tpr"]
    auc_lr = metrics_dict["lr"]["auc"]
    
    fpr_dt, tpr_dt = metrics_dict["dt"]["fpr"], metrics_dict["dt"]["tpr"]
    auc_dt = metrics_dict["dt"]["auc"]

    plt.figure(figsize=(6,5))
    plt.plot(fpr_lr, tpr_lr, label=f"Logística (AUC={auc_lr:.3f})")
    plt.plot(fpr_dt, tpr_dt, label=f"Árvore (AUC={auc_dt:.3f})")
    plt.plot([0,1],[0,1],"--", color="gray", label="Aleatório")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC - Comparação LR vs DT")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Salvar a figura em memória e logar no MLflow
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    mlflow.log_figure(figure=plt.gcf(), artifact_file="curva_roc.png")
    # ou mlflow.log_image(buf, "curva_roc.png") em versões mais recentes.

    plt.close()
