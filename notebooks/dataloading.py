from typing import List#, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# import time, datetime
# stamp_to_ms : Callable[[str]] = lambda T : time.mktime(datetime.datetime.strptime(T, "%Y-%m-%dT%H:%M:%S").timetuple())

def get_labeled_data(df_all_vectors_data:pd.DataFrame):
    # Label Encode so all columns are numerical instead of categorical
    label_encoder = LabelEncoder()
    for category, dtype in zip(df_all_vectors_data, df_all_vectors_data.dtypes):
        if isinstance(dtype, np.dtypes.ObjectDType):
            df_all_vectors_data[category] = label_encoder.fit_transform(df_all_vectors_data[category])

    return df_all_vectors_data

def unique_columns_only(dataframe:pd.DataFrame):
    # Assuming df is your DataFrame
    # Drop columns with only one unique value
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            dataframe.drop(column, axis=1, inplace=True)
    return dataframe

def split_target_and_data(df_all_vectors_data:pd.DataFrame, target_column:str="Type") -> List[pd.DataFrame]:
    # DataFrame containing all columns except 'target_column'
    data = df_all_vectors_data.drop(columns=[target_column])

    # DataFrame containing only column 'target_column'
    target = df_all_vectors_data[[target_column]]

    return data, target

def get_all_data(path_all_vectors:str, test_size:float=.2, unique:bool=True) -> List[np.ndarray]:

    raw_data = pd.read_csv(path_all_vectors)
    df_all_vectors_data = get_labeled_data(raw_data)
    if unique:
        df_all_vectors_data = unique_columns_only(df_all_vectors_data)
    data, target = split_target_and_data(df_all_vectors_data)

    data = data.to_numpy()
    target = target.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size)
    y_train = y_train.squeeze(1)
    y_test = y_test.squeeze(1)

    return X_train, X_test, y_train, y_test
