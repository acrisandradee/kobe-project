# src/kobe_project/pipelines/preparacao_dados/pipeline.py
from kedro.pipeline import Pipeline, node
import mlflow
from .nodes import preparar_dados, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=_pipeline_com_mlflow,
            inputs="dataset_kobe_dev",
            outputs=["base_train", "base_test"],
            name="pipeline_preparacao_dados_node"
        )
    ])

def _pipeline_com_mlflow(df_raw):
    with mlflow.start_run(run_name="PreparacaoDados", nested=True):
        mlflow.set_tag("mlflow.runName", "PreparacaoDados")
        mlflow.set_tag("pipeline", "preparacao_dados")
        mlflow.set_tag("autor", "Cristina")

        # Executar os nodes normalmente
        df_filtered = preparar_dados(df_raw)
        df_train, df_test = split_data(df_filtered)

        return df_train, df_test
