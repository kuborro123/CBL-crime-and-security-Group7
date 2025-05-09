from Data_loader import data_loader

def dataset2():
    query = '''SELECT LSOA_code, count(Crime_type) as count
    FROM crimes
    where Crime_type = 'burglary'
    group by LSOA_code
    
    '''
    return data_loader(query)

def query_Test1():
    query = '''SELECT COUNT(DISTINCT LSOA_code) as count
        FROM crimes
        where Crime_type = 'burglary'
        '''
    return data_loader(query)


print(dataset2())
# print(query_Test1())