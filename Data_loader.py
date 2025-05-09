import pymysql
import pandas as pd

def data_loader(query):
    """
    Loads all the data into a pandas dataframe from the SQL database so it can be used.
    """

    mydb = pymysql.connect(
            host="localhost",
            user="root",
            password="Data_challenge1",
            database="crime_database"
        )

    mycursor = mydb.cursor()

    mycursor.execute(query)

    myresult = mycursor.fetchall()

    data = pd.DataFrame(myresult, columns=[col[0] for col in mycursor.description])
    # data = data.set_index('Crime ID').T.to_dict('list')
    return data


