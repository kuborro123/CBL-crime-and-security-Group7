from pathlib import Path
import csv
import pandas as pd
import mysql.connector



mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Your_Password",
        database="Database_Name"
    )

def path_finder(path):
    """
    :param path a file path to the folder with the data.
    """

    df_location = pd.DataFrame()

    # gets and opens all the files in the folder
    folder_path = Path(path)
    for folder in folder_path.iterdir():
        for file_path in folder.iterdir():
            if file_path.is_file():
                temp_location = file_opener(file_path)
                df_location = pd.concat([df_location, temp_location], axis=0)

    # df_location.dropna(subset=['Crime ID','Latitude'], inplace=True)

    return df_location



def file_opener(path):
    path = str(path)

    df_location = pd.read_csv(path)
    database_maker(df_location)
    return df_location


def database_maker(df_location):
    df_list = df_location.values.tolist()
    for list in df_list:
        mycursor = mydb.cursor()
        data_to_insert = (str(list[0]), str(list[1]), str(list[2]), str(list[3]),
                          str(list[4]), str(list[5]), str(list[6]), str(list[7]),
                          str(list[8]), str(list[9]), str(list[10]), str(list[11]),
                          str(list[12])
                          )
        insert_query = '''INSERT INTO database.usertable 
                                      (id, name, location, verified, followers_count, friends_count, lang) 
                                      VALUES (%s, %s, %s, %s, %s, %s, %s);
        '''
    try:
        mycursor.execute(insert_query, data_to_insert)
    except mysql.connector.errors.IntegrityError:
        pass
    mydb.commit()

df_location = path_finder(r"data/london_crime_database_incomplete")

print(f'len df location: {len(df_location)}')
print(f'location columns: {df_location.columns}')
print(f'NaN values ID: {len(df_location[df_location["Crime ID"].isna()])}')
print(f'NaN latitude: {len(df_location[df_location["Latitude"].isna()])}')
print(f'NaN longitude: {len(df_location[df_location["Longitude"].isna()])}')
print(f'NaN longitude and ID: {len(df_location[df_location["Longitude"].isna() & df_location["Crime ID"].isna()])}')
print(f'burglaries:: {len(df_location[df_location["Crime type"] == "Burglary"])}')

