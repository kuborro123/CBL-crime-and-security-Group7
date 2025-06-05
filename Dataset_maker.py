from Data_loader import data_loader


def burglaries_month_LSOA():
    """"
    selects the amount of burglaries that happend that month in every LSOA
    """
    # Define the query.
    query = '''SELECT LSOA_code, month, count(Crime_type) as crime_count
    FROM crimes
    where (Crime_type = 'burglary')
    group by LSOA_code, month
    
    '''
    return data_loader(query)


def get_deprivation_score():
    """
    selects right deprivation score with LSOA code
    """
    # Define the query.
    query = '''SELECT FeatureCode as LSOA_code, avg(value) as deprivation
        FROM deprivation
        where Measurement = 'Decile '
        group by FeatureCode
        '''
    return data_loader(query)


def burglaries_LSOA():
    """"
    selects the amount of burglaries that happend in every LSOA
    """
    # Define the query.
    query = '''SELECT LSOA_code, count(Crime_type) as crime_count
    FROM crimes
    where (Crime_type = 'burglary')
    group by LSOA_code
    
    '''
    return data_loader(query)


def get_all_burglary_data():
    """
    selects all burglary data with LSOA code
    """
    # Define the query.
    query = '''SELECT *
    FROM crimes
    where (Crime_type = 'burglary')
    '''
    return data_loader(query)


def burglaries_month_LSOA_complete():
    """"
    selects the amount of burglaries that happend that month in every LSOA but for different table
    """
    # Define the query.
    query = '''
    SELECT LSOA_code, month, count(Crime_type) as crime_count
    FROM crimes_complete
    where (Crime_type = 'burglary')
    group by LSOA_code, month
    '''
    return data_loader(query)

def total_burglaries_month():
    '''
    returns the total amount of burglaries per month
    '''
    query = '''
    SELECT month, count(Crime_type) as crime_count
    FROM crimes
    where (Crime_type = 'burglary')
    group by month
    '''
    return data_loader(query)