import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
from typing import List

import neet.data_sources.preprocessing_functions as preproc
import neet.data_sources.constants as constants
from neet.data_sources.schema import get_schema
from neet.constants import DATA_DEVELOP_RAW_PATH, DatasetType
from neet.data_sources.local import read_datasets

os.chdir(DATA_DEVELOP_RAW_PATH)
cwd = Path.cwd()    
      

def attendance_preprocess(dfs:List[pd.DataFrame]) -> pd.DataFrame:
    dfs = preproc.canonicalize_attendance(dfs, constants.COLUMNS_ATTENDANCE)
    attendance = preproc.merge_dfs(dfs, ignore_dtypes=True)
    attendance = preproc.resolve_duplicates_attendance(attendance)
    attendance = preproc.split_and_join_attendance(attendance)
    return attendance

def census_preprocess(dfs:List[pd.DataFrame]) -> pd.DataFrame:
    df = preproc.merge_dfs(dfs, ignore_dtypes=True)
    
    census = (
        df.pipe(preproc.drop_empty_census_rows)
        .pipe(preproc.tidy_sen_cols)
        .pipe(preproc.compare_year_actual_to_year)
        .pipe(preproc.handle_duplicates)
        .pipe(preproc.census_aggregate)
        .pipe(preproc.add_prefix_to_column_names, "census", constants.EXCLUDE_COLUMN_RENAME)
    )
    return census

def nccis_preprocess(dfs:List[pd.DataFrame]) -> pd.DataFrame:
    dfs = preproc.canonicalize_nccis(dfs, constants.COLUMNS_NCCIS)
    nccis = preproc.merge_dfs(dfs, ignore_dtypes=True)
    nccis = preproc.clean_nccis(nccis)
    nccis = preproc.nccis_preprocessing_for_each_model(nccis)
    nccis = preproc.add_prefix_to_column_names(nccis,"nccis",constants.EXCLUDE_COLUMN_RENAME)
    return nccis

def september_guarantee_preprocess(dfs) -> pd.DataFrame:
    dfs = preproc.canonicalize_nccis(dfs, constants.COLUMNS_SEPTEMBER_GUARANTEE)
    september_guarantee = preproc.merge_dfs(dfs, ignore_dtypes=True)
    september_guarantee = preproc.clean_nccis(september_guarantee)
    september_guarantee = preproc.nccis_preprocessing_for_each_model(september_guarantee)
    september_guarantee = preproc.add_prefix_to_column_names(september_guarantee,"september_guarantee",constants.EXCLUDE_COLUMN_RENAME)
    #september_guarantee = september_guarantee.rename(columns = {'nccis_code':'september_guarantee_nccis_code'})
    return september_guarantee

#def ks2_preprocess(dfs:List[pd.DataFrame]) -> pd.DataFrame:
#    dfs = preproc.canonicalize_ks2(dfs, constants.COLUMNS_KS2)
#    ks2 = preproc.merge_dfs(dfs, ignore_dtypes=True)
#    ks2 = preproc.resolve_duplicates_ks2(ks2)
#    ks2 = preproc.add_prefix_to_column_names(ks2,"ks2",constants.EXCLUDE_COLUMN_RENAME)
#    return ks2

def ks4_preprocess(dfs:List[pd.DataFrame]) -> pd.DataFrame:
    dfs = preproc.canonicalize_ks4(dfs, constants.COLUMNS_KS4)
    ks4 = preproc.merge_dfs(dfs, ignore_dtypes=True)
    ks4 = preproc.resolve_duplicates_ks4(ks4)
    ks4 = preproc.add_prefix_to_column_names(ks4,"ks4",constants.EXCLUDE_COLUMN_RENAME)
    return ks4

def exclude_preprocess(dfs:List[pd.DataFrame]) -> pd.DataFrame:
    excluded = preproc.merge_dfs(dfs)
    excluded = preproc.remove_duplicates(excluded)
    excluded = preproc.add_prefix_to_column_names(excluded,"excluded",constants.EXCLUDE_COLUMN_RENAME)
    return excluded

def regional_data_preprocess() -> pd.DataFrame:
    dfcsv = preproc.read_csv_files(cwd/"regional_data", constants.CSV_FILES_REGIONAL_DATA)
    dfexcel = preproc.read_excels(cwd/"regional_data", constants.EXCEL_FILES_REGIONAL_DATA)
    dfs = dfexcel+dfcsv
    pcd_to_lsoa_regional_scores= preproc.filter_and_preprocess_regional_data(dfs,"Bradford",constants.COLUMNS_REGIONAL)
    return pcd_to_lsoa_regional_scores

def school_performance_preprocess(dfs) -> pd.DataFrame:
    school_performance  = preproc.column_mapping_school_performance(dfs, constants.COLUMNS_SCHOOL_PERFORMANCE)
    return school_performance

def postcodes_preprocess() -> pd.DataFrame:
    dfs = preproc.read_csv_files(cwd/"regional_data", constants.CSV_FILES_POSTCODES)
    postcodes = preproc.canonicalize_postcodes(dfs,constants.COLUMNS_POSTCODES)
    return postcodes

def preprocess_all_data(datasets:DatasetType, model_type:str) -> pd.DataFrame:   
    attendance = attendance_preprocess(datasets.attendance)
    census = census_preprocess(datasets.census)
    nccis = nccis_preprocess(datasets.nccis)
    #ks2 = ks2_preprocess()
    ks4 = ks4_preprocess(datasets.ks4)
    excluded = exclude_preprocess(datasets.exclusions)
    # school_performance = school_performance_preprocess(datasets.school_performance)
    pcd_to_lsoa_regional_scores = regional_data_preprocess()
    postcodes = postcodes_preprocess()

    if model_type == "model1" :
        september_guarantee = september_guarantee_preprocess(datasets.september_guarantee)
        # join_df = preproc.joins_training_model1(nccis,attendance,excluded,census,
        # ks4 ,pcd_to_lsoa_regional_scores,
        # september_guarantee,school_performance,postcodes)
        join_df = preproc.joins_training_model1(nccis,attendance,excluded,census,
        ks4 ,pcd_to_lsoa_regional_scores,
        september_guarantee,postcodes)
    elif model_type == "model2":
        # join_df = preproc.joins_training_model2(nccis,attendance,excluded,
        # census,ks4,pcd_to_lsoa_regional_scores,
        # school_performance,postcodes)
        join_df = preproc.joins_training_model2(nccis,attendance,excluded,
        census,ks4,pcd_to_lsoa_regional_scores,postcodes)
    return join_df


model1_df = preprocess_all_data(read_datasets(), "model1")
print("model1",model1_df.shape)
model2_df= preprocess_all_data(read_datasets(), "model2")
print("model2",model2_df.shape)

# write the files to the location
model1_df.to_parquet(os.getenv("INTERMEDIATE_PREPROC_MODEL1"), index=False)
model2_df.to_parquet(os.getenv("INTERMEDIATE_PREPROC_MODEL2"), index=False)

model1_df.to_csv(os.getenv("CSV_INTERMEDIATE_PREPROC_MODEL1"), index=False, header = True)
model2_df.to_csv(os.getenv("CSV_INTERMEDIATE_PREPROC_MODEL2"), index=False, header = True)