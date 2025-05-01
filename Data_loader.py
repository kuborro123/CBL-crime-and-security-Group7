import mysql.connector
import pandas as pd

"""
Loads all the data into a dictionary from the SQL database so it can be used.
"""

mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Your_Password",
        database="Database_Name"
    )

mycursor = mydb.cursor()

mycursor.execute("""
SELECT * 
FROM database.table
""")

myresult = mycursor.fetchall()

data = pd.DataFrame(myresult, columns=[col[0] for col in mycursor.description])
data = data.set_index('id').T.to_dict('list')
