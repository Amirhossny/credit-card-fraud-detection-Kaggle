import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer


def load_data(path):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    return df

def feature_engineering(df):
    df = df.copy()
    
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Day'] = (df['Time'] // (3600 * 24))
    
    return df

def prepare_data(config, base_dir):
    data_path = base_dir / config["data"]["path"]
    target_col = config["data"]["target"]

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_data(data_path)
    df = feature_engineering(df)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return split_data(X, y)

def split_data(x,y, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    x_train2, x_val, y_train2, y_val = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train,random_state=random_state
    )
    return {
        "train": (x_train2, y_train2),
        "val": (x_val, y_val),
        "test": (x_test, y_test)
    }

def preprocessing():
    cols = ['Amount', 'Time', 'Hour', 'Day']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), cols)
        ],
        remainder='passthrough'
    )    
    return preprocessor

def clip_outliers_iqr(x, factor=1.5):
    x = x.copy()
    for col in x.columns:
        q1 = x[col].quantile(0.25)
        q3 = x[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        x[col] = x[col].clip(lower, upper)
    return x

def build_clipping_transformer(factor=1.5):
    return FunctionTransformer(
        clip_outliers_iqr,
        kw_args={'factor': factor},
        validate=False
    )



      















