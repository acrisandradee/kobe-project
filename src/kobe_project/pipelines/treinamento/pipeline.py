from kedro.pipeline import Pipeline, node, pipeline
from .nodes import treinar_logistica, treinar_arvore, avaliar_modelo

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=treinar_logistica,
            inputs=["base_train", "base_test"],
            outputs="predicoes_logistica",
            name="treinar_logistica_node"
        ),
        node(
            func=treinar_arvore,
            inputs=["base_train", "base_test"],
            outputs="predicoes_arvore",
            name="treinar_arvore_node"
        ),
        node(
            func=avaliar_modelo,
            inputs="predicoes_arvore",  # ou "predicoes_logistica", se quiser avaliar o outro
            outputs=None,
            name="avaliar_modelo_node"
        ),
    ])
