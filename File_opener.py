from pathlib import Path
import csv
import pandas as pd


def path_finder(path):
    """
    :param path a file path to the folder with the data.
    """

    df_location = pd.DataFrame()
    df_outcomes = pd.DataFrame()
    df_stop_search = pd.DataFrame()
    # gets and opens all the files in the folder
    folder_path = Path(path)
    for folder in folder_path.iterdir():
        for file_path in folder.iterdir():
            if file_path.is_file():
                temp_location, temp_outcome, temp_stop_search = file_opener(file_path)
                if not temp_location.empty:
                    df_location = pd.concat([df_location, temp_location], axis=0)
                if not temp_outcome.empty:
                    df_outcomes = pd.concat([df_outcomes, temp_outcome], axis=0)
                if not temp_stop_search.empty:
                    df_stop_search = pd.concat([df_stop_search, temp_stop_search], axis=0)

    return df_location, df_outcomes, df_stop_search



def file_opener(path):
    path = str(path)
    df_location = pd.DataFrame()
    df_outcomes = pd.DataFrame()
    df_stop_search = pd.DataFrame()

    if 'street' in path:
        df_location = pd.read_csv(path)
        return df_location, df_outcomes, df_stop_search
    elif 'outcomes' in path:
        df_outcomes = pd.read_csv(path)
        return df_location, df_outcomes, df_stop_search
    else:
        df_stop_search = pd.read_csv(path)
        return df_location, df_outcomes, df_stop_search


df_location, df_outcomes, df_stop_search = path_finder(r"data/London_crime_dataset_incomplete")
