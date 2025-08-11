import numpy as np
import pandas as pd
import neet.data_sources.constants as constants
from scipy.stats import zscore
from pandas.api.types import is_numeric_dtype

# Mean imputations
def mean_value_imputations(df:pd.DataFrame, columns_to_impute:list) -> pd.DataFrame:
    for col in columns_to_impute:
        df[col]=df[col].fillna(df[col].mean())
    return df

#def map_n_to_0_and_y_to_1(dataframe:pd.DataFrame, columns:list, model_type:str) -> pd.DataFrame:
#    mapping = {'N': 0, 'Y': 1}
#    
#    for column in columns:
#        if column in dataframe.columns:
#            try: 
#                dataframe[column] = dataframe[column].astype('category')
#                dataframe[column] = dataframe[column].fillna(dataframe[column].mode()[0])
#                dataframe[column] = dataframe[column].map(mapping)
#                if model_type == "model1":
#                    dataframe[column] = dataframe[column].astype('int')                              
#            except KeyError:
#                # If the column doesn't exist, skip to the next column
#                pass 
#    return dataframe



#def split_and_impute_train_data(df:pd.DataFrame,columns_to_impute:list) -> pd.DataFrame:
#    dataframe_nccis_1 = df[df['nccis_status'] == 1]
#    # For attendance Mean Imputations
#    dataframe_nccis_1 = mean_value_imputations(dataframe_nccis_1, columns_to_impute)
#
#    dataframe_nccis_0 = df[df['nccis_status'] == 0]
#    # For attendance Mean Imputations
#    dataframe_nccis_0 = mean_value_imputations(dataframe_nccis_0, columns_to_impute)
#
#    #df_merged = dataframe_nccis_1.append(dataframe_nccis_0, ignore_index=True)
#    df_merged = pd.concat([dataframe_nccis_1,dataframe_nccis_0])
#    return df_merged


# Attendance feature extraction :
# Define a function to calculate the percentage
def calculate_percentage(unauthorised_absence, authorised_absence, possible_sessions):  
    percentage_unauthorised = (unauthorised_absence / possible_sessions) * 100
    percentage_authorised = (authorised_absence / possible_sessions) * 100
    
    percentage_unauthorised = percentage_unauthorised.replace([np.inf,-np.inf],0)
    percentage_authorised = percentage_authorised.replace([np.inf,-np.inf],0)

    return percentage_unauthorised, percentage_authorised

def fill_missing_with_median(df, columns_to_fill):
    filled_df = df.copy()
    for col in columns_to_fill:
        row_medians = df[columns_to_fill].median(axis=1, skipna=True)
        filled_df[col] = df[col].fillna(row_medians)
    return filled_df

def compute_and_store_median(dataframe, columns_to_compute, new_column_name):
    # Compute the median for the specified columns along the rows
    medians = dataframe[columns_to_compute].median(axis=1)
    
    # Add the computed medians as a new column in the DataFrame
    dataframe[new_column_name] = medians

def get_column_name(df, base_name, suffix):
    """
    Tries to retrieve the column name with a given base name and suffix.
    Falls back to a `.0` suffix if the primary column is not found.
    """
    col = f'{base_name}_{suffix}'
    if col in df.columns:
        return col

    fallback_col = f'{base_name}_{suffix}.0'
    if fallback_col in df.columns:
        return fallback_col

    # Raise error if neither column exists
    raise ValueError(f"Missing column: {base_name}_{suffix} (or {base_name}_{suffix}.0)")

def feature_extract_attendance (df:pd.DataFrame) -> pd.DataFrame:    
    # Calculate the percentages for each suffix digit
    year = ['8', '9', '10', '11']

    for suffix in year:
        unauth_col = get_column_name(df, 'unauthorised_absence', suffix)
        auth_col = get_column_name(df, 'authorised_absence', suffix)
        poss_col = get_column_name(df, 'possible_sessions', suffix)
        
        df[f'unauth_percentage_{suffix}'], df[f'auth_percentage_{suffix}'] = calculate_percentage(df[unauth_col], df[auth_col], df[poss_col])
    
    columns_to_impute_auth = [
        "auth_percentage_8",
        "auth_percentage_9",
        "auth_percentage_10",
        "auth_percentage_11",
    ]
    columns_to_impute_unauth = [
        "unauth_percentage_8",
        "unauth_percentage_9",
        "unauth_percentage_10",
        "unauth_percentage_11",
    ]
    
    columns_to_impute = columns_to_impute_auth + columns_to_impute_unauth


    df = fill_missing_with_median(df, columns_to_impute_auth)
    df = fill_missing_with_median(df, columns_to_impute_unauth)
    
    #df = mean_value_imputations(df, columns_to_impute)
    
    df = df.drop(columns=constants.COLUMNS_TO_IMPUTE_MEAN_ATTENDANCE)
    
    df['median_unauthorised_absences'] = df[columns_to_impute_unauth].median(axis=1)
    df['median_unauthorised_absences'] = df['median_unauthorised_absences'].fillna(df['median_unauthorised_absences'].median())
    
    df['median_authorised_absences'] = df[columns_to_impute_auth].median(axis=1)
    df['median_authorised_absences'] = df['median_authorised_absences'].fillna(df['median_authorised_absences'].median())
    
    df = df.drop(columns = columns_to_impute)

    return df

def calculate_sen_need_level(row):
    senneed1 = row['census_senneed1']
    senneed2 = row['census_senneed2']
    if (not pd.isna(senneed1) and senneed1 != 0) and (pd.isna(senneed2) or senneed2 == 0):
        return 1
    elif (not pd.isna(senneed1) and senneed1 != 0) and (not pd.isna(senneed2) and senneed2 != 0):
        return 2
    elif senneed1 == 0 and senneed2 == 0:
        return 0
    else:
        return 0

def feature_extract_census(df:pd.DataFrame) -> pd.DataFrame:
    #df['sen_need_level'] = df.apply(calculate_sen_need_level, axis=1)
    df['sen_need_level'] = df['census_senneed1'] + df['census_senneed2']
    df = df.drop(columns=['census_senneed1','census_senneed2'])
    df = df.drop(columns=['census_senneed1_y11','census_senneed2_y11']) 
    df['census_senprovision_y11'] = df['census_senprovision_y11'].replace({'S':'E'})
    return df

def map_special_school(value:str) -> int:
    if value is not None and 'special' in value:
        return False
    else:
        return True

def feature_extract_school_performance(df:pd.DataFrame) -> pd.DataFrame:
    df['special_school_flag'] = df['schooltype_y'].apply(map_special_school).astype(bool)
    df['schooltype_y'] = df['schooltype_y'].fillna(df['schooltype_y'].mode()).astype('category')
    #df= df.drop(columns=['schooltype_y'])

    rating_mapping = {
    'outstanding': 5,
    'good': 4,
    'requires improvement': 3,
    'inadequate': 2,
    'serious weaknesses': 1,
    'special measures': 0
    }

    # Convert to lowercase and map the values
    #df['ofstedrating'] = df['ofstedrating'].fillna(df['ofstedrating'].mode()[0])
    df['ofstedrating'] = df['ofstedrating'].str.lower().map(rating_mapping).astype('category')
    return df

def ks4_priorband_ptq_ee_transform(s:pd.Series) -> pd.Series:
    """
    Replace 4 with np.Nan, because 4 in column "KS4_PRIORBAND_PTQ_EE" 
    is equal to NULL.

    https://find-npd-data.education.gov.uk/en/data_elements/f0601e6f-ad26-474e-9573-c72428c0b692
    
    Args:
        pd.Series for "ks4_priorband_ptq_ee"
    
    Returns:
        pd.Series with transformed data
    """
    return s.replace(4, np.NaN).astype("Int8")

def ks4_interactions(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculates interaction effects for our ks4 predictors.
    
    Args:
        df pd.DataFrame with columns "ks4_priorband_ptq_ee", "ks4_att8", "ks4_pass_94"
    
    Returns:
        pd.DataFrame with additional interaction columns
    """
    df["ks4_priorband_ptq_ee_ks4_att8"] =  df["ks4_priorband_ptq_ee"] * df["ks4_att8"]
    df["ks4_priorband_ptq_ee_ks4_pass_94"]=  df["ks4_priorband_ptq_ee"] * df["ks4_pass_94"] 
    df["ks4_att8_ks4_pass_94"] = df["ks4_att8"] * df["ks4_pass_94"]
    #df = df.drop(columns = ["ks4_att8" , "ks4_pass_94", "ks4_priorband_ptq_ee"])
    
    return df

# need to edit this below function
def feature_extract_ks4(df:pd.DataFrame) -> pd.DataFrame:
    df['ks4_priorband_ptq_ee'] = df["ks4_priorband_ptq_ee"].fillna(0)
    df['ks4_priorband_ptq_ee'] = df["ks4_priorband_ptq_ee"].apply(lambda x : 0 if x == 4 else x)
    df['ks4_priorband_ptq_ee'] = df['ks4_priorband_ptq_ee'].astype(int)
    #df = ks4_interactions(df)
    return df


def normalize_the_numeric_val(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            if np.issubdtype(df[col].dtype, np.number):
                unique_values = df[col].nunique()
                if unique_values > 2:
                    df[col] = zscore(df[col])
                else:
                    pass
            else:
                pass
    return df   