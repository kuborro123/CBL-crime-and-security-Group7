from pathlib import Path
import csv
import pandas as pd
import pymysql



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
                

    df_location.dropna(subset=['Crime ID','Latitude'], inplace=True)

    database_maker(df_location)
    return df_location



def file_opener(path):
    path = str(path)

    df = pd.read_csv(path)

    return df


def database_maker(df_location):
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="Data_challenge2",
        database="crime_database"
    )

    mycursor = mydb.cursor()

    insert_query = '''
        INSERT INTO crimes 
        (`Crime ID`, `Month`, `Reported by`, `Falls within`, `Longitude`, `Latitude`,
         `Location`, `LSOA code`, `LSOA name`, `Crime type`, `Last outcome category`, `Context`)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    '''

    df_list = df_location.values.tolist()

    for row in df_list:
        try:
            data_to_insert = tuple(str(item) for item in row)
            mycursor.execute(insert_query, data_to_insert)
        except pymysql.MySQLError as e:
            print(f"Error inserting row: {e}")
            continue  # Skip to next row

    mydb.commit()
    mycursor.close()
    mydb.close()

def deprivation(path):
    df_deprivation = file_opener(path)

    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="Data_challenge1",
        database="crime_database"
    )

    mycursor = mydb.cursor()

    insert_query = '''
            INSERT INTO deprivation 
            (FeatureCode,DateCode,Measurement,Units,Value,Indices_of_Deprivation)
            VALUES (%s, %s, %s, %s, %s, %s);
        '''

    df_list = df_deprivation.values.tolist()

    for row in df_list:
        try:
            data_to_insert = tuple(str(item) for item in row)
            mycursor.execute(insert_query, data_to_insert)
        except pymysql.MySQLError as e:
            print(f"Error inserting row: {e}")
            continue  # Skip to next row

    mydb.commit()
    mycursor.close()
    mydb.close()

df_location = path_finder(r"dataset")
deprivation(r'data/imd2019lsoa.csv')

print(f'len df location: {len(df_location)}')
print(f'location columns: {df_location.columns}')
print(f'NaN values ID: {len(df_location[df_location["Crime ID"].isna()])}')
print(f'NaN latitude: {len(df_location[df_location["Latitude"].isna()])}')
print(f'NaN longitude: {len(df_location[df_location["Longitude"].isna()])}')
print(f'NaN longitude and ID: {len(df_location[df_location["Longitude"].isna() & df_location["Crime ID"].isna()])}')
print(f'burglaries:: {len(df_location[df_location["Crime type"] == "Burglary"])}')

