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

    # Log no MLflow
    mlflow.log_metric("linhas_filtradas", df.shape[0])
    mlflow.log_metric("colunas_filtradas", df.shape[1])

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

    # Log no MLflow
    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("train_rows", df_train.shape[0])
    mlflow.log_metric("test_rows", df_test.shape[0])

    proporcao = y_train.value_counts(normalize=True).to_dict()
    mlflow.log_metric("prop_0_train", proporcao.get(0.0, 0))
    mlflow.log_metric("prop_1_train", proporcao.get(1.0, 0))

    return df_train, df_test