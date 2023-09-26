from pathlib import Path
import numpy as np
import pandas as pd
from typing import Literal
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
#import pandas_profiling as pdp
import neet.data_sources.feature_extraction_functions as f
import neet.data_sources.datacleaning_functions as dcf
import neet.data_sources.constants as constants
# Get .env data
load_dotenv()

def data_imputations(df:pd.DataFrame, model_type:str) ->pd.DataFrame:
    
    if model_type == 'model1':
        #Read File
        df = dcf.mapping_target_varible(df)
        df = dcf.encode_and_data_type_conversions(df,constants.COLUMNS_CATEGORICAL_NOMINAL_MODEL1, 
                                                     constants.COLUMNS_CATEGORICAL_ORDINAL_MODEL1,
                                                     constants.COLUMNS_NUMERIC_MODEL1)
        df['distance_from_home_to_school_km'] = df.apply(dcf.calculate_distance, axis=1)
        #This is temporary way
        columns_to_drop = ['home_latitude','home_longitude',
                            'school_latitude','school_longitude',
                            'september_guarantee_academic_age', 'census_surname','census_forename',
                            'september_guarantee_order_of_nccis_update','census_cohort', 'ks4_cohort',
                            'september_guarantee_confirmed_date', 
                            'census_estab','ks4_estab', 'excluded_year',
                            'excluded_cohort','school_postcode', 
                            'lsoa_name_2011','postcode','september_guarantee_time_recorded',
                            'september_guarantee_parent','ks4_pass_94','census_ethnicity','census_gender']
        df = df.drop(columns=columns_to_drop) 

    elif model_type == 'model2':
        #Read File
        #df= pd.read_parquet(os.getenv("INTERMEDIATE_PREPROC_MODEL2"),engine='pyarrow')
        df = dcf.mapping_target_varible(df)
        df = dcf.encode_and_data_type_conversions(df,constants.COLUMNS_CATEGORICAL_NOMINAL_MODEL2, 
                                                     constants.COLUMNS_CATEGORICAL_ORDINAL_MODEL2,
                                                     constants.COLUMNS_NUMERIC_MODEL2)
        df['distance_from_home_to_school_km'] = df.apply(dcf.calculate_distance, axis=1)
        columns_to_drop = ['home_latitude','home_longitude',
                            'school_latitude','school_longitude',
                            'nccis_academic_age', 'census_surname','census_forename',
                            'nccis_order_of_nccis_update','census_cohort',
                            'ks4_cohort', 
                            'nccis_confirmed_date',
                            'census_estab','ks4_estab', 'excluded_year',
                            'excluded_cohort','school_postcode', 
                            'lsoa_name_2011','postcode','nccis_time_recorded',
                            'nccis_parent','ks4_pass_94','census_ethnicity','census_gender']
        df = df.drop(columns=columns_to_drop) 
    
    X = df.drop(columns=["nccis_status"])
    y = df["nccis_status"]
    # Split the data into train and test sets

    X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)


    return X_train_model, X_test_model, y_train_model, y_test_model


def post_split_feature_extraction_training_data(X_train_model:pd.DataFrame,y_train_model:pd.Series) -> pd.DataFrame:
    df = X_train_model
    df = f.feature_extract_attendance(df)
    df = f.feature_extract_census(df)
    df = f.feature_extract_school_performance(df)
    df = f.feature_extract_ks4(df)
    df = pd.concat([df, y_train_model],axis=1)
    return df


def post_split_feature_extraction_testing_data(X_test_model:pd.DataFrame,y_test_model:pd.Series,model_type:str) -> pd.DataFrame:
    df = X_test_model
    df = f.feature_extract_attendance(df)
    df = f.feature_extract_census(df)
    df = f.feature_extract_school_performance(df)
    df = f.feature_extract_ks4(df)
    df = pd.concat([df, y_test_model],axis=1)
    return df
