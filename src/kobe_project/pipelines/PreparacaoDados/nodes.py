import pandas as pd
from sklearn.model_selection import train_test_split

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas = [
        "lat", "lon", "minutes_remaining", "period",
        "playoffs", "shot_distance", "shot_made_flag"
    ]
    return df[colunas].dropna()

def split_data(df: pd.DataFrame):
    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["shot_made_flag"],
        random_state=42
    )
    return train, test
