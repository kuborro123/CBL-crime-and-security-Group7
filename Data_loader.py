import pymysql
import pandas as pd

def data_loader(query):
    """
    Loads all the data into a pandas dataframe from the SQL database so it can be used.
    :param query: A query to get from the database.
    """

    # Connect to the database.
    mydb = pymysql.connect(
            host="localhost",
            user="root",
            password="Data_challenge1",
            database="crime_database"
        )

    # Make a cursor and run the query.
    mycursor = mydb.cursor()
    mycursor.execute(query)
    myresult = mycursor.fetchall()

    # Put data in a dataframe, can also be done in a dictionary if you see the commented line.
    data = pd.DataFrame(myresult, columns=[col[0] for col in mycursor.description])
    # data = data.set_index('Crime ID').T.to_dict('list')
    return data


