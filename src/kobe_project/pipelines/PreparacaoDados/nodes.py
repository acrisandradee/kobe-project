import pandas as pd
from typing import Tuple
import pandas as pd
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
    
    print(f"Dimensão df_train: {df_train.shape}")
    print(f"Dimensão df_test:  {df_test.shape}")
    print("\nProporção de classes (treino):\n", df_train["shot_made_flag"].value_counts(normalize=True))
    print("\nProporção de classes (teste):\n", df_test["shot_made_flag"].value_counts(normalize=True))

    return df_train, df_test