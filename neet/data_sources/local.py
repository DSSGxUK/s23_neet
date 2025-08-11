import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
from typing import List

import neet.data_sources.preprocessing_functions as preproc
import neet.data_sources.constants as constants
from neet.data_sources.schema import get_schema
from neet.constants import DATA_DEVELOP_RAW_PATH, DatasetType

os.chdir(DATA_DEVELOP_RAW_PATH)
cwd = Path.cwd()


def read_attendance() -> List[pd.DataFrame]:
    schema = get_schema("attendance")
    dfs = preproc.read_excels(
        cwd /"attendance", constants.EXCEL_FILES_ATTENDANCE, schema
    )
    return dfs


def read_census() -> List[pd.DataFrame]:
    schema = get_schema("census")
    dfs = preproc.read_excels(
        cwd /"census", constants.EXCEL_FILES_CENSUS, schema
    )
    return dfs


def read_exclusions() -> List[pd.DataFrame]:
    schema = get_schema("exclusions")
    dfs = preproc.read_excels(
        cwd /"attendance", constants.EXCEL_FILE_EXCLUSIONS, schema
    )
    return dfs


def read_ks2() -> List[pd.DataFrame]:
    schema = get_schema("ks2")
    dfs = preproc.read_excels(
        cwd /"ks2", constants.EXCEL_FILES_KS2, schema
    )
    return dfs


def read_ks4() -> List[pd.DataFrame]:
    schema = get_schema("ks4")
    dfs = preproc.read_excels(
        cwd/"ks4", constants.EXCEL_FILES_KS4, schema
    )
    return dfs


def read_nccis() -> List[pd.DataFrame]:
    schema = get_schema("nccis")
    folder = cwd /"nccis"
    dfs = preproc.read_excels(folder, constants.EXCEL_FILES_NCCIS, schema)
    return dfs


def read_school_performance() -> List[pd.DataFrame]:
    schema = get_schema("school-performance")
    dfs = preproc.read_excels(cwd/"school_performance", constants.EXCEL_FILES_SCHOOL_PERFORMANCE, schema)
    return dfs


def read_september_guarantee() -> List[pd.DataFrame]:
    schema = get_schema("nccis")
    dfs = preproc.read_csv_files(
        cwd /"september_nccis", constants.CSV_FILES_SEPTEMBER_GUARANTEE, schema
    )
    return dfs

#def read_regional_data() -> List[pd.DataFrame]:
#    dfcsv = preproc.read_csv_files(cwd/"raw"/"regional_data", constants.CSV_FILES_REGIONAL_DATA)
#    dfexcel = preproc.read_excels(cwd/"raw"/"regional_data", constants.EXCEL_FILES_REGIONAL_DATA)
#    dfs = dfexcel+dfcsv
#    return dfs

#def read_postcodes() -> List[pd.DataFrame]:
#    dfs = preproc.read_csv_files(cwd/"raw"/"regional_data", constants.CSV_FILES_POSTCODES)
#    return dfs

def read_datasets() -> DatasetType:
    """
    Reads all the data from disk based on fixed folder paths and names.

    Returns:
        DatasetType: namedtuple with dataframes in alphabetical order
    """
    attendance = read_attendance()
    census = read_census()
    exclusions = read_exclusions()
    ks4 = read_ks4()
    september_guarantee = read_september_guarantee()
    nccis = read_nccis()
    # school_performance = read_school_performance()
    #regional_data = read_regional_data()
    #postcodes = read_postcodes()

    return DatasetType(
        attendance,
        census,
        exclusions,
        ks4,
        nccis,
        september_guarantee,
        # school_performance
        #regional_data,
        #postcodes
    )
