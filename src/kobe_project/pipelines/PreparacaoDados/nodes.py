import mlflow
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas = [
        "lat",
        "lon",               
        "minutes_remaining",
        "period",
        "playoffs",
        "shot_distance",
        "shot_made_flag"
    ]
    
    df = df[colunas]
    df = df.dropna(subset=colunas)
    
    print("Dimensão final:", df.shape)
    return df

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop("shot_made_flag", axis=1)
    y = df["shot_made_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  
    )

    df_train = X_train.copy()
    df_train["shot_made_flag"] = y_train
    
    df_test = X_test.copy()
    df_test["shot_made_flag"] = y_test
    
    return df_train, df_test


def pipeline_preparacao_dados(
    input_path: str,
    train_path: str,
    test_path: str,
    test_size: float = 0.2,
):
  

    with mlflow.start_run(run_name="PreparacaoDados"):
        df_raw = pd.read_parquet(input_path)
        df_filtered = preparar_dados(df_raw)

        df_train, df_test = split_data(df_filtered, test_size=test_size, random_state=42)

        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("linhas_filtered", df_filtered.shape[0])
        mlflow.log_metric("train_rows", df_train.shape[0])
        mlflow.log_metric("test_rows", df_test.shape[0])

        proporcao_treino = df_train["shot_made_flag"].value_counts(normalize=True).to_dict()
        mlflow.log_metric("prop_class0_treino", proporcao_treino.get(0.0, 0))
        mlflow.log_metric("prop_class1_treino", proporcao_treino.get(1.0, 0))
        Path(train_path).parent.mkdir(parents=True, exist_ok=True)  
        df_train.to_parquet(train_path, index=False)

        Path(test_path).parent.mkdir(parents=True, exist_ok=True)
        df_test.to_parquet(test_path, index=False)

        print("[INFO] Pipeline finalizado:")
        print(" - Treino salvo em:", train_path)
        print(" - Teste salvo em :", test_path)


if __name__ == "__main__":
    # Ajuste os caminhos conforme sua estrutura
    input_path = "/Data/raw/dataset_kobe_dev.parquet"
    train_path = "/Data/processed/base_train.parquet"
    test_path = "/Data/processed/base_test.parquet"

    pipeline_preparacao_dados(
        input_path=input_path,
        train_path=train_path,
        test_path=test_path,
        test_size=0.2,
    )
