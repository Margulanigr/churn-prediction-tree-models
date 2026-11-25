from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Путь строится от расположения этого файла
# notebook/data_prep.py -> корень проекта -> data/raw/Bank_Churn.csv
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Bank_Churn.csv"

ID_COLUMN = 'customer_id'
TARGET = 'churn'
CATEGORICAL_COLUMNS = ["country", "gender"]

def prepare_data(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    df = pd.read_csv(csv_path)

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    cat_cols = ["country", "gender"]
    encoder = OneHotEncoder(sparse_output=False, drop="first")

    encoded_train = encoder.fit_transform(X_train[cat_cols])
    encoded_test = encoder.transform(X_test[cat_cols])

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    encoded_df_train = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
    encoded_df_test = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)

    X_train_final = pd.concat([X_train.drop(columns=cat_cols), encoded_df_train], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=cat_cols), encoded_df_test], axis=1)

    return X_train_final, X_test_final, y_train, y_test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"Пропусков: {X_train.isna().sum().sum()} (train), {X_test.isna().sum().sum()} (test)")
    print(f"Столбцы совпадают: {list(X_train.columns) == list(X_test.columns)}")