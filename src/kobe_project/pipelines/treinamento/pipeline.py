# src/kobe_project/pipelines/treinamento/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import configurar_pycaret, treinar_modelos_e_avaliar, plotar_roc
import mlflow

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=_pipeline_treinamento,
            inputs=["base_train", "base_test"],  # do catalog.yml
            outputs=None,
            name="pipeline_treinamento_node"
        ),
    ])

def _pipeline_treinamento(df_train, df_test):
    # Abre um run do MLflow
    with mlflow.start_run(run_name="Treinamento"):
        # 1) Configura PyCaret
        configurar_pycaret(df_train, target_col="shot_made_flag", random_state=42)

        # 2) Treina modelos e avalia
        metrics_dict = treinar_modelos_e_avaliar(df_test)

        # 3) Plota a curva ROC
        plotar_roc(metrics_dict)
