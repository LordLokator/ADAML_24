# import time, datetime
# from typing import Callable
# stamp_to_ms : Callable[[str]] = lambda T : time.mktime(datetime.datetime.strptime(T, "%Y-%m-%dT%H:%M:%S").timetuple())

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd

def get_labeled_data(df_all_vectors_data:pd.DataFrame, label_encoder = None):
    # Label Encode so all columns are numerical instead of categorical
    label_encoder = LabelEncoder()
    for category, dtype in zip(df_all_vectors_data, df_all_vectors_data.dtypes):
        if isinstance(dtype, np.dtypes.ObjectDType):
            df_all_vectors_data[category] = label_encoder.fit_transform(df_all_vectors_data[category])

    return df_all_vectors_data, label_encoder

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
    df_all_vectors_data, _ = get_labeled_data(raw_data)
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


def standard_scale(df, numerical_cols, scaler = None):
    # Create a new StandardScaler instance and fit it on the original data
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[numerical_cols])

    # Inverse transform the numerical columns
    scaled_numerical_cols = scaler.transform(df[numerical_cols])

    # Replace the transformed numerical columns with the original values
    df[numerical_cols] = scaled_numerical_cols
    
    return df, scaler

def fix_attack_codes(df, train_attack_codes, other_cols):
    # get the attack codes in the df
    df_attack_codes = [c for c in df.columns if c not in other_cols]
    attack_codes_to_delete = [ac for ac in df_attack_codes if ac not in train_attack_codes]
    attack_codes_to_add = [ac for ac in train_attack_codes if ac not in df_attack_codes]
    
    # add missing attack cols
    for col in attack_codes_to_add:
        df[col] = 0
        
    # add other col
    df['other_attack_codes'] = df[attack_codes_to_delete].any(axis=1).astype(int)

    # drop these columns
    df.drop(columns=attack_codes_to_delete, inplace=True)

    
    return df

def merge_df(attack_df, vector_df):
    merged_df = pd.merge(vector_df, attack_df[['Attack ID','Start time', 'End time', 'Type']], on='Attack ID', how='left')
    return merged_df

def get_attack_duration(df):
    # convert date columns
    df['Start time'] = pd.to_datetime(df['Start time'].replace('0', np.nan), errors = 'coerce')
    df['End time'] = pd.to_datetime(df['End time'].replace('0', np.nan), errors = 'coerce')

    # drop NaN values
    df = df.dropna(subset=['Start time', 'End time'])
    df = df.reset_index(drop=True)

    # get duration
    df['Attack duration'] = np.abs(df['End time'] - df['Start time'])
    df['Attack duration'] = df['Attack duration'].apply(lambda x: x.total_seconds())

    #drop the start and end column
    df = df.drop(columns=['Start time', 'End time'], errors='ignore')

    return df.copy()

def encode_attack_labels(df):
    # convert string to list
    df['attack code list'] = df['Attack code'].apply(lambda x: x.replace(', ',',').split(','))
    # Identify unique labels
    unique_labels = set(label for sublist in df['attack code list'] for label in sublist)

    # create empty one-hot encoded columns
    for label in unique_labels:
        df[label] = 0

    # iterate through rows and update one-hot encoded columns
    for idx, row in df.iterrows():
        labels = row['attack code list']
        for label in labels:
            df.at[idx, label] = 1

    # drop the original 'attack code list' column
    df.drop('attack code list', axis=1, inplace=True)

    return df.copy()

def convert_victim_ip(df):
    # ip address
    df['victim IP num'] = df['Victim IP'].apply(lambda x: int(x.split('_')[1]))
    return df.copy()

def convert_time(df):
    # time column -> is_weekday, time_of_day
    # convert date string to datetime
    df.rename(columns = {'Time':'time string'}, inplace = True)

    # was the time on the weekend
    df['time'] = pd.to_datetime(df['time string'])
    df['is_weekday'] = df['time'].apply(lambda x: int(x.weekday() < 5))
    df['is_weekday']

    # time of day in seconds
    df['time_of_day'] = df['time'].dt.time
    df['time_of_day'] = df['time_of_day'].apply(lambda x: 60*60*x.hour + 60*x.minute + x.second)

    return df.copy() 

def preprocces_for_aug(df):
    # attack duration
    df = get_attack_duration(df)
    # attack labels
    df = encode_attack_labels(df)
    # victim ip to num
    df = convert_victim_ip(df)
    # time col -> is_weekday, time_of_day
    df = convert_time(df)
    return df

def preprocess(vector_df, attack_df):
    # add labels to the vectors df
    vector_df = merge_df(attack_df, vector_df)
    
    vector_df = preprocces_for_aug(vector_df)
    
    # drop columns
    cols_to_drop = ['Card', 'Attack ID', 'Detect count', 'Victim IP', 'Attack code', 'time string', 'time']
    vector_df = vector_df.drop(cols_to_drop, axis=1)

    return vector_df.copy()