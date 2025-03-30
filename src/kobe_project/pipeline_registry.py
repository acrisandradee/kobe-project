from kobe_project.pipelines import PreparacaoDados, treinamento

def register_pipelines():
    return {
        "preparacao": PreparacaoDados.create_pipeline(),
        "treinamento": treinamento.create_pipeline(),
        "__default__": PreparacaoDados.create_pipeline() + treinamento.create_pipeline(),
    }
