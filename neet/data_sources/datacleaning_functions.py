from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd

# Drop Columns that are more than 60% empty
def drop_empty_columns(df:pd.DataFrame, percentage_threashold:int) -> pd.DataFrame:
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().sum() / len(df) * 100

    # Filter out columns with more than 60% missing data
    columns_to_drop = missing_percentage[missing_percentage > percentage_threashold].index

    # Drop the selected columns from the DataFrame
    return df.drop(columns=columns_to_drop)


def encode_and_data_type_conversions(df:pd.DataFrame,columns_categorical_nominal:list,columns_categorical_ordinal:list,columns_numeric:list) -> pd.DataFrame:
    columns_to_encode = columns_categorical_nominal + columns_categorical_ordinal
    #convert to numeric
    for col in columns_numeric:
        df[col] = df[col].astype(int)

    #Write conversions here
    for col in columns_to_encode: 
        df[col] = df[col].astype('category')

    return df

def mapping_target_varible(df:pd.DataFrame) -> pd.DataFrame:
    df["nccis_status"] = np.where(df["nccis_code"].isin([540,610,615,616,619,620,630,640,650,660,670,680,710]), 1, 0) # neet_status
    df = df[~df['nccis_code'].isin([810,820,830])]
    df= df.drop(columns='nccis_code')
    #df['nccis_code']=df['nccis_code'].astype('category')
    
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371

    lat_1_rad, lon_1_rad = np.radians(lat1), np.radians(lon1)
    lat_2_rad, lon_2_rad = np.radians(lat2), np.radians(lon2)

    dlon_rad = lon_2_rad - lon_1_rad
    dlat_rad = lat_2_rad - lat_1_rad

    a = np.sin(dlat_rad / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon_rad / 2.0) ** 2
    haversine_rad = 2 * np.arcsin(np.sqrt(a))
    haversine_km = haversine_rad * earth_radius

    return haversine_km

def calculate_distance(row):
    # Get the latitude and longitude for the home postcode
    lat1, lon1 = row['home_latitude'], row['home_longitude']

    # Get the latitude and longitude for the school postcode
    lat2, lon2 = row['school_latitude'], row['school_longitude']

    # Calculate the Haversine distance between the two points
    distance = haversine_distance(lat1, lon1, lat2, lon2)

    return distance


