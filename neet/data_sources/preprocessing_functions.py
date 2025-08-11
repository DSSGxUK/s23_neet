from pathlib import Path
import re
from typing import List
import numpy as np
import pandas as pd
import pandera as pa
from neet.data_sources.schema import get_schema


# READ DATA
# ************************************************************************************
def read_excels(folder: Path, files_and_sheets: dict, schema: pa.DataFrameSchema | None = None) -> list[pd.DataFrame]:
    """
    Reads all Excel files in a specifc folder if filename and sheets are in 
    "files_and_sheets" and add them to a list of dataframes.

    Args:
        folder: pathlib.Path to the folder 
        files_and_sheets: nested dict with information on each excel file 
                        { filename:{"sheets":[0], "cohort":["2018-19"], "years":["11"]} }
        schema: pandera schema for this dataset_type

    Returns:
        a list of pandas dataframes
    """

    dfs = []
    for file in folder.iterdir():
        if file.name in files_and_sheets.keys():
            for i, sheet in enumerate(files_and_sheets[file.name]["sheets"]):
                df = read_excel(folder=folder,
                                filename=file.name,
                                sheetname=sheet,
                                cohort=str(
                                    files_and_sheets[file.name]["cohort"][i]),
                                year=int(
                                    files_and_sheets[file.name]["years"][i]),
                                schema=schema)
                dfs.append(df)
    return dfs


def read_excel(folder: Path, filename: str, sheetname: str, cohort: str, year: int, schema: pa.DataFrameSchema | None = None) -> pd.DataFrame:
    """
    Wrapper to read an Excel file, add the academic year and drop empty rows/columns.
    Uses pandas.read_excel() under the hood.

    Args:
        folder: pathlib Path() to the folder where file searched.
        filename: name of the file to read with the extenstion (only tested for .xlsx)
        sheetname: The sheetname to read. See pandas read_excel() for possible values
        acad_year: A string of the academic year, e.g. "2018-19"
        year: Integer of the year a student is in e.g. "11" for Y11.
        schema: optional schema to add

    Returns:
        A pandas dataframe
    """
    try:
        df = pd.read_excel(folder / filename, sheet_name=sheetname)
    except ValueError as e:
        xls = pd.ExcelFile(folder / filename)
        print(f"Worksheets found in {filename} in {folder} are {xls.sheet_names}.")
        raise ValueError(f"Error reading {filename} in {folder}: {e}")

    # Column names to snake case (also transforms simple camelCase and removes special characters):
    df.columns = (df.columns
                  .str.replace(' ', '_', regex=False)
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace('[', '', regex=False)
                  .str.replace(']', '', regex=False)
                  .str.replace('/', '_', regex=False)
                  .str.replace('\\', '_', regex=False)
                  .str.replace('?', '', regex=False)
                  .str.replace(',', '', regex=False)
                  .str.replace('(?<=[a-z])(?=[A-Z])', '_', regex=True)
                  .str.lower()
                  )

    # Validate the schema
    # if 'ncyear_actual' in df.columns:
    #     df['ncyear_actual'].replace('N1', np.nan, inplace=True)
    
    # if 'support_level' in df.columns:
    #     df['support_level'].replace('Z', np.nan, inplace=True)
    
    # if schema:
    #     try:
    #         df = schema.validate(df, lazy=True)
    #     except pa.errors.SchemaErrors as err:
    #         err.failure_cases  # dataframe of schema errors
    #         err.data  # invalid dataframe
    #         # raise
    #         pass
    

    df['cohort'] = cohort

    # Sanity check for the year
    df['year'] = year if 7 <= year <= 13 else np.NaN

    # for col in df.select_dtypes(include=['datetime64[ns]']).columns:
    # df[col] = df[col].astype(str)

    return df

# For CSV FILES Read
def read_csv_files(folder: Path, files_and_sheets: dict, schema: pa.DataFrameSchema | None = None) -> List[pd.DataFrame]:
    """
    Reads all Excel files in a specifc folder if filename and sheets are in 
    "files_and_sheets" and add them to a list of dataframes.

    Args:
        folder: pathlib.Path to the folder 
        files_and_sheets: nested dict with information on each excel file 
                        { filename:{"sheets":[0], "cohort":["2018-19"], "years":["11"]} }
        schema: optional pandera schema for the datasettype

    Returns:
        a list of pandas dataframes
    """

    dfs = []
    for file in folder.iterdir():
        if file.name in files_and_sheets.keys():
            for i, sheet in enumerate(files_and_sheets[file.name]["sheets"]):
                df = read_csv_file(filepath_or_buffer=folder / file.name,
                                   cohort=str(
                                       files_and_sheets[file.name]["cohort"][i]),
                                   year=int(
                                       files_and_sheets[file.name]["years"][i]),
                                   schema=schema)

                dfs.append(df)
    return dfs


def read_csv_file(filepath_or_buffer: Path, cohort: str, year: int, schema: pa.DataFrameSchema | None = None) -> pd.DataFrame:
    """
    Wrapper to read an Excel file, add the academic year and drop empty rows/columns.
    Uses pandas.read_excel() under the hood.

    Args:
        filepath_or_buffer: Same as pd.read_csv() but limited to path
        cohort: A string with the cohort, e. g. "2018-19"
        year: Integer of the year a student is in e.g. "11" for Y11.
        schema: optional pandera schema

    Return:
        A pandas dataframe

    Raises:
        pa.errors.SchemaError if df does not match the schema
    """
    df = pd.read_csv(filepath_or_buffer, low_memory=False)

    # Column names to snake case (also transforms simple camelCase and removes special characters):
    df.columns = (df.columns
                  .str.replace(' ', '_', regex=False)
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace('[', '', regex=False)
                  .str.replace(']', '', regex=False)
                  .str.replace('/', '_', regex=False)
                  .str.replace('\\', '_', regex=False)
                  .str.replace('?', '', regex=False)
                  .str.replace(',', '', regex=False)
                  .str.replace('(?<=[a-z])(?=[A-Z])', '_', regex=True)
                  .str.lower()
                  )

    # Validate the schema
    if schema:
        try:
            df = schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            err.failure_cases  # dataframe of schema errors
            err.data  # invalid dataframe
            raise

    df['cohort'] = str(cohort)

    # Sanity check for the year
    df['year'] = year if 7 <= year <= 13 else np.NaN

    return df


def merge_dfs(dfs: list[pd.DataFrame], ignore_dtypes: bool = False) -> pd.DataFrame:
    """
    Takes a list of dfs with the same header an merges them.
    Column names for the returned dataset are transformed to snake-case.

    Args:
        dfs: A list of pandas dataframes to merge
        ignore_dtypes: Skips the dtype check for all columns

    Return: 
        A merged pandas dataframe 

    Raises:
        ValueError: If the column names or dtypes do not match
    """
    # if not all(set(df.columns) == set(dfs[0].columns) for df in dfs):
    #     raise ValueError(
    #         "Dataframes need to have the same column names before merging.")

    if not ignore_dtypes:
        if not all(set(df.dtypes) == set(dfs[0].dtypes) for df in dfs):
            raise ValueError(
                "All columns need to have the same dtype before merging.")

    df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # Drop empty rows, but ignore the cohort, year and uid columns which will always have a value.
    df = df.dropna(how='all', subset=[col for col in df.columns if col not in [
                   'cohort', 'stud_id', 'year']])

    return df


def add_prefix_to_column_names(dataframe: pd.DataFrame, prefix: str, exclude_columns: list) -> pd.DataFrame:
    if prefix == 'ks4':
        dataframe.rename(columns={'cohort': 'ks4_cohort'}, inplace=True)
    elif prefix == 'attendance':
        dataframe.rename(columns={'cohort': 'attendance_cohort'}, inplace=True)
        dataframe.rename(columns={'year': 'attendance_year'}, inplace=True)
    else:
        new_columns = []
        for column in dataframe.columns:
            if column in exclude_columns:
                new_columns.append(column)
            else:
                new_columns.append(prefix + '_' + column)
        dataframe.columns = new_columns
    return dataframe

# ************************************************************************************
# Functions for processing ATTENDANCE DATA
def canonicalize_attendance(dfs: list[pd.DataFrame], columns_to_keep: set) -> list[pd.DataFrame]:

    # Do not drop these columns by accident
    columns_to_keep.update({'cohort', 'stud_id'})

    columns_to_keep = list(columns_to_keep)

    for idx, df in enumerate(dfs):

        dfs[idx] = df[columns_to_keep]

    return dfs


def resolve_duplicates_attendance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the sum of a row if more than one row per person per academic year
    is available. Means that a student was a two different schools for the term.

    TODO: Replace the dirty hack!
    """

    # Dirty hack to remove problematic id.
    # generalize to other Councils
    df = df.drop(df[df['stud_id'].isin([471807, 534073])].index)

    # Removes all the duplicates by getting the mean and reseting the index after
    # We should take a sum, but we have data quality issues with students that are enrolled in different schools
    # or where the school name changed.
    df = (
        df.groupby(["stud_id", "year"])
        .sum(numeric_only=True)
        .reset_index()
    )
    return df


def split_and_join_attendance(attendance: pd.DataFrame) -> pd.DataFrame:
    """
    This function splits the attendance dataframe into different dataframes based on year and 
    then joins the year 11 dataframe with others to get a complete profile of single student across multiple years

    Args: 
        pandas df with attendance data 

    Return: 
        Attendance data with only year 11 
    """

    # Step 1: Split the dataframe into different dataframes based on year
    year_dataframes = {}

    for year in attendance['year'].unique():
        year_dataframes[year] = attendance[attendance['year'] == year]

    # Step 2: Join the dataframes based on year == 11
    result_df = year_dataframes[11]

    for year, df in year_dataframes.items():
        if year == 11 or year < 7:
            continue  # Skip the year == 11 dataframe, as it's the base dataframe

        # Left join other dataframes to the year == 11 dataframe
        result_df = result_df.merge(
            df, on='stud_id', how='left', suffixes=('', f'_{year}'))

    # Step 3: Cleaner dataset
    columns_name_mapping = {
        "possible_sessions": "possible_sessions_11",
        "attendance_count": "attendance_count_11",
        "authorised_absence": "authorised_absence_11",
        "unauthorised_absence":  "unauthorised_absence_11"
    }

    result_df = result_df.rename(columns=columns_name_mapping, errors='ignore')

    return result_df.drop(["year", "year_7", "year_8", "year_9", "year_10"], axis=1)

# ************************************************************************************
# Functions for processing CENSUS DATA
def tidy_sen_cols(df: pd.DataFrame) -> pd.DataFrame:
    df[df.loc[:, 'senneed1'] == 0] = np.nan
    df[df.loc[:, 'senneed2'] == 0] = np.nan
    # df['senprovision'] = ['E' if val == 'S' else val for val in df['senprovision']]
    # df.loc[:,'senprovision'] = df['senprovision'].replace({'S':'E'})
    return df


def drop_empty_census_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Drop empty rows, but ignore the "'cohort', 'stud_id', 'year'" and uid column which will always have a value.#
    ignore_cols = ['cohort', 'forename', 'surname',
                   'stud_id', 'year', 'date_of_birth']
    df = df.dropna(how='all', subset=[
                   col for col in df.columns if col not in ignore_cols])
    return df


def compare_year_actual_to_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares the year and the year actual. If they are not equal it drops the student.
    """
    sliced = df[["stud_id", "year", "ncyear_actual"]]

    # df["year"] should never be empty.
    mask = (
        sliced["ncyear_actual"].isnull()
        | sliced["year"].isnull()
        | (sliced["ncyear_actual"] == sliced["year"])
    )

    sliced["compare"] = mask

    uids_to_drop = sliced.loc[sliced["compare"] == False, "stud_id"].to_list()

    # Drop based on uids
    return df[~df["stud_id"].isin(uids_to_drop)]


def handle_duplicates(df: pd.DataFrame):
    """
    Detects if we have multiple entries for the same student.
    Keeps the entry with less NaNs. If equal first entry is kept.
    """
    df["nan_count"] = df.isna().sum(axis=1)

    df = df.sort_values(by=["stud_id", "year", "nan_count"])

    # Drop duplicate rows keeping the first occurrence
    cleaned_df = df.drop_duplicates(subset=["stud_id", "year"], keep="first")

    return cleaned_df.drop(columns=["nan_count"])


def census_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the data so we have one column per school year.
    """

    def agg_mode(x: pd.Series):
        """Takes the most frequent value. If frequency is the same selects randomly"""

        # Check for series series with NaN:
        if x.isnull().all():
            return np.NaN

        x = x.mode()

        return x if len(x) == 1 else np.random.choice(x)

    def entry_year_changed(x: pd.Series):
        """
        Compares the Y11 estab with the etab a student spent most time at between
        year 7 and 11. If they are different the student spent most time not at their
        GCSE school.
        """

        # Check for series series with NaN:
        if x.isnull().all():
            return np.NaN

        x = x.astype("string")

        pattern = re.compile(r"(20[0-3]\d)")

        x = x.str.findall(pattern).str[0]
        x = x.nunique()

        return x - 1 if x > 0 else np.NaN

    # Get year 11 SEN data to add is back to the data after aggregation for the dashboard.
    y11_sen = df.loc[df["year"] == 11, [
        "stud_id", "senprovision", "senneed1", "senneed2"]]
    y11_sen = y11_sen.set_index("stud_id").rename(
        columns={
            "senprovision": "senprovision_y11",
            "senneed1": "senneed1_y11",
            "senneed2": "senneed2_y11",
        }
    )

    # Functions to aggregate different columns
    agg_funcs = {
        # "stud_id": "first",
        "forename": "first",
        "surname": "first",
        "entry_date": entry_year_changed,
        "estab": "first",
        "gender": "first",
        "ethnicity": agg_mode,
        "language": agg_mode,
        "senneed1": "count",
        "senneed2": "count",
        "senunit_indicator": "sum",
        "resourced_provision_indicator": "sum",
        "fsme_on_census_day": "sum",
        "cohort": "first",
    }

    # Sort so aggregate by "first" in year 11
    df = df.sort_values("year", ascending=False).groupby(
        "stud_id").agg(agg_funcs)

    # Add year 11 sen data back to the frame
    df = df.join(y11_sen, on="stud_id")

    # Rename entry_date to no_school_changed
    df = df.rename(columns={"entry_date": "school_changed_count"})

    df['resourced_provision_indicator'] = df['resourced_provision_indicator'].astype(
        int)
    df['fsme_on_census_day'] = df['fsme_on_census_day'].astype(int)
    
    # df['ncyear_actual'].replace('N1', np.nan, inplace=True)

    return df.reset_index()


# ************************************************************************************
# Functions for processing EXCLUSION DATA
def canonicalize_excluded(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:

    return dfs


def remove_duplicates(df) -> pd.DataFrame:
    """
    Just drops duplicate entries and keeps the last one.
    Not a good solution ...
    """

    return df.drop_duplicates("stud_id", keep="last")

# ************************************************************************************
# Functions for processing KS2 DATA


def canonicalize_ks2(dfs: list[pd.DataFrame], columns_to_keep: set) -> list[pd.DataFrame]:

    # Do not drop these columns by accident
    columns_to_keep.update({'cohort', 'stud_id'})

    columns_to_keep = list(columns_to_keep)
    for idx, df in enumerate(dfs):

        dfs[idx] = df[columns_to_keep]

    return dfs


def resolve_duplicates_ks2(df: pd.DataFrame) -> pd.DataFrame:

    # Add a helper column with the NaN count
    df['nan_count'] = df.iloc[:, 1:].apply(lambda x: sum(pd.isnull(x)), axis=1)

    # Sort by NaN count and drop the duplicates.
    df = df.sort_values(
        'nan_count', ascending=False).drop_duplicates('stud_id')

    # Remove the helper column.
    df_cleaned = df.drop('nan_count', axis=1)

    return df_cleaned

# ************************************************************************************
# Functions for processing KS4 Data


def canonicalize_ks4(dfs: list[pd.DataFrame], columns_to_keep: set) -> list[pd.DataFrame]:

    # Do not drop these columns by accident
    columns_to_keep.update({'cohort', 'stud_id'})

    columns_to_keep = list(columns_to_keep)

    for idx, df in enumerate(dfs):

        dfs[idx] = df[columns_to_keep]
    # Update U= ungraded and NAN's as 0
    # for df in dfs:
    #    df['ks4_apeng_91'] = df['ks4_apeng_91'].apply(lambda x: 0 if (x == 'U') or (x is None) else x)
    #    df['ks4_hgmath_91'] = df['ks4_hgmath_91'].apply(lambda x: 0 if (x == 'U') or (x is None) else x)

    return dfs


def resolve_duplicates_ks4(df: pd.DataFrame) -> pd.DataFrame:

    # Add a helper column with the NaN count
    df['nan_count'] = df.iloc[:, 1:].apply(lambda x: sum(pd.isnull(x)), axis=1)

    # Sort by NaN count and drop the duplicates.
    df = df.sort_values(
        'nan_count', ascending=False).drop_duplicates('stud_id')

    # Remove the helper column.
    return df.drop('nan_count', axis=1)


# ************************************************************************************
# Functions for processing REGIONAL DATA
def filter_and_preprocess_regional_data(df: pd.DataFrame, council_name: str, columns_to_keep: set) -> pd.DataFrame:

    # Assign the dataframes
    regional_scores = df[0]
    pcd_to_lsoa = df[1]

    regional_scores.columns
    # For Regional Scores DataFrame
    # regional_scores = regional_scores[columns_to_keep]
    regional_scores.rename(
        columns={'lsoa_code_2011': 'lsoa_code'}, inplace=True)
    regional_scores = regional_scores[regional_scores['local_authority_district_name_2019'] == council_name]
    # print(regional_scores)

    # For PCD to LSOA DataFrame

    pcd_to_lsoa.rename(columns={'lsoa11cd': 'lsoa_code'}, inplace=True)
    pcd_to_lsoa = pcd_to_lsoa[pcd_to_lsoa['ladnm'] == council_name]
    # print(pcd_to_lsoa)

    pcd_to_lsoa_regional_scores = pd.merge(
        pcd_to_lsoa, regional_scores, how='left', on='lsoa_code')

    # columns_to_drop = ['lsoa_code','ladcd','lsoa11nm','msoa11nm','ladnm','ladnmw','cohort',
    #                   'year','lsoa_name_2011','local_authority_district_name_2019','pcd7',
    #                   'pcd8','dointr','doterm','usertype','oa11cd','msoa11cd']

    # pcd_to_lsoa_regional_scores=pcd_to_lsoa_regional_scores.drop(columns_to_drop,axis=1)

    pcd_to_lsoa_regional_scores.rename(
        columns={'pcds': 'postcode'}, inplace=True)
    pcd_to_lsoa_regional_scores = pcd_to_lsoa_regional_scores[columns_to_keep]

    return pcd_to_lsoa_regional_scores

# ************************************************************************************
# Functions for processing NCCIS


def canonicalize_nccis(dfs: list[pd.DataFrame], columns_to_keep: set) -> list[pd.DataFrame]:

    # Do not drop these columns by accident
    columns_to_keep.update({'time_recorded', 'stud_id'})

    # Cohort is the wrong name for the column so we change it to
    columns_name_mapping = {'student_id': 'stud_id', 'cohort': 'time_recorded'}

    datetime_cols = ['confirmed_date']

    for df in dfs:
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
            except:
                try:
                    df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")
                except:
                    try:
                        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y")
                    except:
                        df[col] = pd.to_datetime(
                            df[col], format="%d/%m/%Y %H:%M")
            # print(df[col].dtype)

            df[col] = df[col].dt.strftime('%Y-%m-%d')
            df[col] = pd.to_datetime(df[col], errors='coerce')

    columns_to_keep = list(columns_to_keep)
    for idx, df in enumerate(dfs):

        # Rename the student_id column for one of the datasets
        frame = df.rename(columns=columns_name_mapping, errors='ignore')
        dfs[idx] = frame[columns_to_keep]
        # dfs[idx] = frame[[col for col in columns_to_keep if col in frame.columns]]

    return dfs


def clean_nccis(df: pd.DataFrame) -> pd.DataFrame:
    df['support_level'] = df['support_level'].astype(str)
    # Drop students with acdemic age above 19 and SEND flag
    df = df.drop(df[(df['academic_age'] > 19) & (df['send'] == "Y")].index)

    return df


def nccis_preprocessing_for_each_model(df: pd.DataFrame) -> pd.DataFrame:
    # Sort values
    # df.sort_values(by=['stud_id', 'start_date'], inplace=True)

    # Add a new 'order_of_nccis_update' column based on 'confirmed_date' for each 'stud_id'
    df['order_of_nccis_update'] = df.groupby(
        'stud_id')['confirmed_date'].rank()

    # The 'order_of_nccis_update' column will now contain the rank based on 'confirmed_date' within each 'stud_id' group
    df = df[df['order_of_nccis_update'] == 1]
    return df

# ************************************************************************************
# Functions for REGIONAL DATA
# Functions for regional data enrichment


def filter_and_preprocess_regional_data(df: pd.DataFrame, council_name: str, columns_to_keep: set) -> pd.DataFrame:

    # Assign the dataframes
    regional_scores = df[0]
    pcd_to_lsoa = df[1]

    regional_scores.columns
    # For Regional Scores DataFrame
    regional_scores.rename(
        columns={'lsoa_code_2011': 'lsoa_code'}, inplace=True)
    regional_scores = regional_scores[regional_scores['local_authority_district_name_2019'] == council_name]

    # For PCD to LSOA DataFrame

    pcd_to_lsoa.rename(columns={'lsoa11cd': 'lsoa_code'}, inplace=True)
    pcd_to_lsoa = pcd_to_lsoa[pcd_to_lsoa['ladnm'] == council_name]

    pcd_to_lsoa_regional_scores = pd.merge(
        pcd_to_lsoa, regional_scores, how='left', on='lsoa_code')

    pcd_to_lsoa_regional_scores.rename(
        columns={'pcds': 'postcode'}, inplace=True)
    columns_to_keep = list(columns_to_keep)
    pcd_to_lsoa_regional_scores = pcd_to_lsoa_regional_scores[columns_to_keep]

    return pcd_to_lsoa_regional_scores


# ************************************************************************************
# Functions for School Performance Data
def column_mapping_school_performance(df: pd.DataFrame, columns_to_keep: set) -> pd.DataFrame:
    df = df[0]
    column_name_mapping = {
        'postcode': 'school_postcode',
        'estab': 'ks4_estab',
        'perctot': 'perc_overall_abscence',
        'ppersabs10': 'perc_pers_absentee',
        'nor': 'total_num_of_pupil',
        'pnorg': 'perc_of_girls',
        'pnorb': 'perc_of_boys',
        'psenelse': 'perc_of_sen_pupil',
        'psenelk': 'perc_of_eligible_sen_pupil',
        'numeal': 'english_not_flang',
        'pnumeal': 'perc_english_not_flang',
        'pnumengfl': 'perc_english_flang',
        'pnumuncfl': 'first_lang_unclassified',
        'numfsm': 'no_of_pupil_eligible',
        # 'numfsmever':'num_pf_fsm_6yr',
        'norfsmever': 'total_fsm_ever',
        'pnumfsmever': 'p_no_pf_fsm_6yr'}
    # Rename the columns using the mapping
    df = df.rename(columns=column_name_mapping)
    columns_to_keep = list(columns_to_keep)
    df = df[columns_to_keep]
    return df

# ************************************************************************************
# Functions for DISTANCE FROM SCHOOL
# Calculate Distance from School
def canonicalize_postcodes(df: pd.DataFrame, columns_to_keep: set) -> pd.DataFrame:
    columns_to_keep = list(columns_to_keep)
    df = df[0]
    df = df[columns_to_keep]
    return df


def join_postcodes(df1_merged_df: pd.DataFrame, df2_postcodes: pd.DataFrame) -> pd.DataFrame:
    joined_df = pd.merge(df1_merged_df, df2_postcodes,
                         on='postcode', how='left')
    column_mappings_df = {'latitude': 'home_latitude',
                          'longitude': 'home_longitude'}
    joined_df = joined_df.rename(columns=column_mappings_df)

    column_mappings_postcode = {'postcode': 'school_postcode'}
    df2_postcodes = df2_postcodes.rename(columns=column_mappings_postcode)
    
    # Find the positions of the duplicate columns
    duplicate_positions = [i for i, col in enumerate(joined_df.columns) if col == "school_postcode"]
    # Drop the second occurrence of 'school_postcode'
    if len(duplicate_positions) > 1:
         joined_df = joined_df.loc[:, ~joined_df.columns.duplicated()]

    joined_df = pd.merge(joined_df, df2_postcodes,
                         on='school_postcode', how='left')
    column_mappings_df = {'latitude': 'school_latitude',
                          'longitude': 'school_longitude'}
    joined_df = joined_df.rename(columns=column_mappings_df)
    return joined_df

# ************************************************************************************
# Functions for JOINING DATA
# create joins for model1 and model2 to prepare the data for training purpose
# model 1 : without NCCIS
def joins_training_model1(nccis: pd.DataFrame, attendance: pd.DataFrame,
                          excluded: pd.DataFrame, census: pd.DataFrame, ks4: pd.DataFrame,
                          regional_data: pd.DataFrame, september_guarantee: pd.DataFrame,
                          school_performance: pd.DataFrame, postcodes: pd.DataFrame) -> pd.DataFrame:

    # Create NCCIS subset df
    nccis_sel_col = ['stud_id', 'nccis_code']
    subset_df = nccis.loc[:, nccis_sel_col]

    # Census and NCCIS Inner Join
    census_nccis_inner_joined_df = pd.merge(
        subset_df, census, how='inner', on='stud_id')

    # join the september guarantee data with census nccis data
    census_nccis_septg_joined_df = pd.merge(
        census_nccis_inner_joined_df, september_guarantee, on='stud_id', how='left')

    # join attendance and Exclusions
    attendance_exclusions_left_joined_df = pd.merge(
        attendance, excluded, how='left', on='stud_id')

    # ks2 and ks4 join
    # ks4_ks2_left_joined_df = pd.merge(df6_ks4,df5_ks2,how='left',on='stud_id')

    # join the attednance, exclusions, ks4 and ks2
    ks4_ks2_attendance_exclusions_joined_df = pd.merge(
        ks4, attendance_exclusions_left_joined_df, how='inner', on='stud_id')

    # inner join with census and nccis
    all_df = pd.merge(census_nccis_septg_joined_df,
                      ks4_ks2_attendance_exclusions_joined_df, how='inner', on='stud_id')

    # Regional Data Enrichment
    final_df_regional_enrichment = pd.merge(
        all_df, regional_data, on='postcode', how='left')

    # school data enrichment
    final_df_school_data_enrichment = pd.merge(
        final_df_regional_enrichment, school_performance,  on='ks4_estab', how='left')

    # Enrich the data with lat and long
    final_df = join_postcodes(final_df_school_data_enrichment, postcodes)

    return final_df


def joins_training_model2(nccis: pd.DataFrame, attendance: pd.DataFrame,
                          excluded: pd.DataFrame, census: pd.DataFrame,
                          ks4: pd.DataFrame, regional_data: pd.DataFrame, school_performance: pd.DataFrame,
                          postcodes: pd.DataFrame) -> pd.DataFrame:

    # join census and nccis inner join
    census_nccis_inner_joined_df = pd.merge(
        nccis, census, how='inner', on='stud_id')

    # join attendance and Exclusions
    attendance_exclusions_left_joined_df = pd.merge(
        attendance, excluded, how='left', on='stud_id')

    # ks2 and ks4 join
    # ks4_ks2_left_joined_df = pd.merge(df6_ks4,df5_ks2,how='left',on='stud_id')

    # join the attednance, exclusions, ks4 and ks2
    ks4_ks2_attendance_exclusions_joined_df = pd.merge(
        ks4, attendance_exclusions_left_joined_df, how='inner', on='stud_id')

    # inner join with census and nccis
    all_df = pd.merge(census_nccis_inner_joined_df,
                      ks4_ks2_attendance_exclusions_joined_df, how='inner', on='stud_id')

    # Regional Data Enrichment
    final_df_regional_enrichment = pd.merge(
        all_df, regional_data, on='postcode', how='left')

    # School Performance Data
    final_df_school_data_enrichment = pd.merge(
        final_df_regional_enrichment, school_performance, how='left',  on='ks4_estab')

    # Enrich the data with lat and long
    final_df = join_postcodes(final_df_school_data_enrichment, postcodes)

    return final_df


# Create joins for model1 and model2 to prepare the data for just testing purpose

# def joins_testing_model1_without_nccis(df1_attendance:pd.DataFrame,df2_excluded:pd.DataFrame,df3_census_proc:pd.DataFrame,
#                                       df4_ks2:pd.DataFrame, df5_ks4:pd.DataFrame, df6_regional_data:pd.DataFrame,
#                                       df7_september_guarantee_proc:pd.DataFrame,df8_school_performance:pd.DataFrame,
#                                       df9_postcodes:pd.DataFrame) -> pd.DataFrame:
#
#    #join attendance and Exclusions
#    attendance_exclusions_left_joined_df = pd.merge(df1_attendance,df2_excluded,how='left',on='stud_id')
#
#    #ks2 and ks4 join
#    ks4_ks2_left_joined_df = pd.merge(df5_ks4, df4_ks2,how='left',on='stud_id')
#
#    #join the attednance, exclusions, ks4 and ks2
#    ks4_ks2_attendance_exclusions_joined_df= pd.merge(ks4_ks2_left_joined_df,attendance_exclusions_left_joined_df,how='inner',on='stud_id')
#
#    #inner join with census and nccis
#    census_attendance_attainment_inner_join= pd.merge(df3_census_proc,ks4_ks2_attendance_exclusions_joined_df,how='inner',on='stud_id')
#
#    #left join with september guarantee
#    september_guarantee_enriched_df = pd.merge(census_attendance_attainment_inner_join,df7_september_guarantee_proc,how='left',on='stud_id')
#
#    #Regional Data Enrichment
#    final_df_regional_enrichment = pd.merge(september_guarantee_enriched_df, df6_regional_data, how = 'left', on= 'postcode')
#
#    #School Performance Data
#    final_df_school_data_enrichment = pd.merge(final_df_regional_enrichment, df8_school_performance, how='left', on='ks4_estab')
#
#    #Enrich the data with lat and long
#    final_df = join_postcodes(final_df_school_data_enrichment, df9_postcodes)
#
#    return final_df
#
# def joins_testing_model2_with_nccis(df1_nccis_proc:pd.DataFrame, df2_attendance:pd.DataFrame,df3_excluded:pd.DataFrame,
#                                    df4_census_proc:pd.DataFrame, df5_ks2:pd.DataFrame, df6_ks4:pd.DataFrame,
#                                    df7_regional_data:pd.DataFrame, df8_school_performance:pd.DataFrame,
#                                    df9_postcodes:pd.DataFrame) -> pd.DataFrame:
#    # Name of the columns you want to exclude
#    column_to_exclude = 'nccis_code'
#
#    # Subsetting the dataframe without the specified column
#    df_subset_nccis_proc = df1_nccis_proc.drop(column_to_exclude, axis=1)
#
#    #Census NCCIS join
#    census_nccis_inner_joined_df = pd.merge(df_subset_nccis_proc,df4_census_proc,how='inner',on='stud_id')
#
#    #join attendance and Exclusions
#    attendance_exclusions_left_joined_df = pd.merge(df2_attendance,df3_excluded,how='left',on='stud_id')
#
#    #ks2 and ks4 join
#    ks4_ks2_left_joined_df = pd.merge(df6_ks4,df5_ks2,how='left',on='stud_id')
#
#    #join the attednance, exclusions, ks4 and ks2
#    ks4_ks2_attendance_exclusions_joined_df= pd.merge(ks4_ks2_left_joined_df,attendance_exclusions_left_joined_df,how='inner',on='stud_id')
#
#    #inner join with census and nccis
#    all_df = pd.merge(census_nccis_inner_joined_df,ks4_ks2_attendance_exclusions_joined_df,how='inner',on='stud_id')
#
#    #Regional Data Enrichment
#    final_df_regional_enrichment = pd.merge(all_df,  df7_regional_data, on='postcode',how='left')
#
#    #school performance enrichment
#    final_df_school_data_enrichment = pd.merge(final_df_regional_enrichment, df8_school_performance, how='left', on='ks4_estab')
#
#    #Enrich the data with lat and long
#    final_df = join_postcodes(final_df_school_data_enrichment, df9_postcodes)
#
#    return final_df


# Check Missing IDs
def check_missing_ids(df1_nccis_proc: pd.DataFrame, df2_attendance: pd.DataFrame, df3_ks2: pd.DataFrame, df4_ks4: pd.DataFrame, df5_census_proc: pd.DataFrame, df6_excluded: pd.DataFrame, df7_final_df: pd.DataFrame) -> dict:
    """
    This function helps to identify the student_id's from each dataframe for whom predictions cannot be made due to incomplete data 

    Input: nccis_proc, attendance, ks2, ks4, census_proc, final_df

    output : Dictionary of missing student_ids from each dataframe 
    """
    # Function to check for missing IDs in a given dataframe
    def missing_ids(df, merged_df):
        return df[~df['stud_id'].isin(merged_df['stud_id'])]['stud_id'].tolist()

    # Find missing IDs in each dataframe
    nccis_missing_ids = missing_ids(df1_nccis_proc, df7_final_df)
    attendance_missing_ids = missing_ids(df2_attendance, df7_final_df)
    ks2_missing_ids = missing_ids(df3_ks2, df7_final_df)
    ks4_missing_ids = missing_ids(df4_ks4, df7_final_df)
    census_missing_ids = missing_ids(df5_census_proc, df7_final_df)
    exclusion_missing_ids = missing_ids(df6_excluded, df7_final_df)

    return {
        'nccis_missing_ids': nccis_missing_ids,
        'attendance_missing_ids': attendance_missing_ids,
        'ks2_missing_ids': ks2_missing_ids,
        'ks4_missing_ids': ks4_missing_ids,
        'census_missing_ids': census_missing_ids,
        'exclusion_missing_ids': exclusion_missing_ids
    }
