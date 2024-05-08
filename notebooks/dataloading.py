# import time, datetime
# from typing import Callable
# stamp_to_ms : Callable[[str]] = lambda T : time.mktime(datetime.datetime.strptime(T, "%Y-%m-%dT%H:%M:%S").timetuple())

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd

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

def get_all_data(path_all_vectors:str, test_size:float=.2, unique:bool=True) -> list:

    raw_data = pd.read_csv(path_all_vectors)
    df_all_vectors_data = get_labeled_data(raw_data)
    if unique:
        df_all_vectors_data = unique_columns_only(df_all_vectors_data)
    data_df, target_df = split_target_and_data(df_all_vectors_data)

    info_data = list(data_df.columns.values)
    info_label = list(target_df.columns.values)

    data = data_df.to_numpy()
    target = target_df.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size)
    y_train = y_train.squeeze(1)
    y_test = y_test.squeeze(1)


    return (X_train, X_test, y_train, y_test), (info_data, info_label)

def preprocess(vector_df, attack_df):
    # add labels to the vectors df
    vector_df = pd.merge(vector_df, attack_df[['Attack ID', 'Type']], on='Attack ID', how='left')
    
    # one-hot-encode attack labels
    # convert string to list
    vector_df['attack code list'] = vector_df['Attack code'].apply(lambda x: x.replace(', ',',').split(','))
    # Identify unique labels
    unique_labels = set(label for sublist in vector_df['attack code list'] for label in sublist)

    # create empty one-hot encoded columns
    for label in unique_labels:
        vector_df[label] = 0

    # iterate through rows and update one-hot encoded columns
    for idx, row in vector_df.iterrows():
        labels = row['attack code list']
        for label in labels:
            vector_df.at[idx, label] = 1

    # drop the original 'attack code list' column
    vector_df.drop('attack code list', axis=1, inplace=True)

    # ip address
    vector_df['victim IP num'] = vector_df['Victim IP'].apply(lambda x: int(x.split('_')[1]))

    # time column -> is_weekday, time_of_day
    # convert date string to datetime
    vector_df.rename(columns = {'Time':'time string'}, inplace = True)

    # was the time on the weekend
    vector_df['time'] = pd.to_datetime(vector_df['time string'])
    vector_df['is_weekday'] = vector_df['time'].apply(lambda x: int(x.weekday() < 5))

    # time of day in seconds
    vector_df['time_of_day'] = vector_df['time'].dt.time
    vector_df['time_of_day'] = vector_df['time_of_day'].apply(lambda x: 60*60*x.hour + 60*x.minute + x.second)

    # drop columns
    cols_to_drop = ['Card', 'Attack ID', 'Detect count', 'Victim IP', 'Attack code', 'time string', 'time']
    vector_df = vector_df.drop(cols_to_drop, axis=1)

    return vector_df.copy()

def standard_scale(df, numerical_cols):
    # Create a new StandardScaler instance and fit it on the original data
    scaler = StandardScaler()
    scaler.fit(df[numerical_cols])

    # Inverse transform the numerical columns
    scaled_numerical_cols = scaler.transform(df[numerical_cols])

    # Replace the transformed numerical columns with the original values
    df[numerical_cols] = scaled_numerical_cols
    
    return df