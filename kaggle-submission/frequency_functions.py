import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import  os
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd

###

def load_data(path, verbose=False):
    """
    Load a dataset consisting of multiple CSV files from a directory.

    Parameters:
    - path (str): The directory containing the CSV files.
    - verbose (bool): If True, prints the head and shape of the combined DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing all the data from the CSV files.
    """
    li = []

    # Iterate over each file in the directory
    for filename in os.listdir(path):
        if filename.endswith(".csv"):  # Ensure only CSV files are processed
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path)
            li.append(df)

    # Concatenate all the DataFrames in the list
    output = pd.concat(li, ignore_index=True)

    # Display information if verbose is True
    if verbose:
        print(output.head())
        print(f'The shape of the data is: {output.shape}')

    return output




def add_time_sec(df):
    """
    Adds a 'Seconds' column representing the time elapsed in seconds since the beginning of each match
    and removes the 'Timestamp' column.
    """
    first_timestamps = df.groupby("MatchID")["Timestamp"].min().reset_index(name="FirstTimestamp")
    df = df.merge(first_timestamps, on="MatchID")
    df["Seconds"] = (df["Timestamp"] - df["FirstTimestamp"]) // 1000
    df.drop(columns=["Timestamp", "FirstTimestamp"], inplace=True)
    return df


def add_freq_normalized(df):
    """
    Adds a 'NormalizedFrequency' column to the DataFrame, representing the normalized frequency of tweets at each second"""
    df_matchs = {match_id: group for match_id, group in df.groupby('MatchID')}
    processed_matchs = []
    for match_id, match_df in df_matchs.items():
        total_tweets=match_df.shape[0]
        max_seconds=match_df['Seconds'].max()
        tot_per_sec=match_df.groupby('Seconds').size()
        normalized_frequency= tot_per_sec/total_tweets*max_seconds
        match_df['NormalizedFrequency']=match_df['Seconds'].map(normalized_frequency)
        processed_matchs.append(match_df)
    final_df = pd.concat(processed_matchs, ignore_index=True)
    return final_df



def make_df_for_kaggle(df_bis, pred, threshold=0.55):
    """
    Creates a DataFrame in the format required for the Kaggle submission" by aggegating the labels in each period with a vote"""
    predictions_eval = pd.DataFrame({
        'ID': df_bis['ID'],
        'MatchID': df_bis['MatchID'],
        'PeriodID': df_bis['PeriodID'],
        'Predicted': pred
    })
    predictions_eval['Predicted'] = (predictions_eval['Predicted'] > threshold).astype(int)
    predictions_eval=predictions_eval[['ID', 'Predicted_NN_three']]
    return predictions_eval
    