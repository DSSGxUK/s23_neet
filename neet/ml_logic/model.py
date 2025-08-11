
"""
Custom Models for Predicting Young People at risk of becoming NEET
"""
from typing import Literal
from neet.ml_logic.model_logistic_l1 import *
from neet.ml_logic.model_lightgbm import *
import neet.ml_logic.data_imputations as di
import neet.data_sources.feature_extraction_functions as f
import neet.data_sources.constants as constants
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, SMOTENC
import pickle
from dotenv import load_dotenv
import os
import pkgutil
import io
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

import neet.ml_logic.preprocessor as preprocessor
from neet.data_sources.local import read_datasets
import neet.data_sources.datacleaning_functions as dcf

from pathlib import Path
os.chdir("../..")
cwd = Path.cwd()

# Function to Get data for Logistic Regression
def get_data_for_logistic_regression(df: pd.DataFrame, model_type: Literal["model1", "model2"]) -> tuple:
    X_train_model, X_test_model, y_train_model, y_test_model = di.data_imputations(
        df, model_type)
    train_data_model = di.post_split_feature_extraction_training_data(
        X_train_model, y_train_model)
    test_data_model = di.post_split_feature_extraction_testing_data(
        X_test_model, y_test_model, model_type)

    train_data_model = train_data_model.dropna()
    test_data_model = test_data_model.dropna()

    train_stud_ids = pd.DataFrame(train_data_model['stud_id'])
    test_stud_ids = pd.DataFrame(test_data_model['stud_id'])

    train_data_model = train_data_model.drop(columns=['stud_id'])
    test_data_model = test_data_model.drop(columns=['stud_id'])

    train_data_model = f.normalize_the_numeric_val(train_data_model)
    test_data_model = f.normalize_the_numeric_val(test_data_model)

    X_train_model = train_data_model.drop(columns='nccis_status')
    y_train_model = train_data_model['nccis_status']
    X_test_model = test_data_model.drop(columns='nccis_status')
    y_test_model = test_data_model['nccis_status']

    # print(X_train_model.shape)
    # print(y_train_model.shape)
    # print(X_test_model.shape)
    # print(y_test_model.shape)

    # drop all the categorical features
    categorical_columns_train = X_train_model.select_dtypes(
        include=['category']).columns
    categorical_columns_test = X_test_model.select_dtypes(
        include=['category']).columns

    X_train_model = X_train_model.drop(columns=categorical_columns_train)
    X_test_model = X_test_model.drop(columns=categorical_columns_test)

    return X_train_model, X_test_model, y_train_model, y_test_model


# Function to run logistic regression model
def run_logistic_regression_model(df: pd.DataFrame, model_type: Literal["model1", "model2"]):
    X_train_model, X_test_model, y_train_model, y_test_model = get_data_for_logistic_regression(
        df, model_type)
    # Create instance
    logistic_model = LogisticRegressionL1()

    # create pipeline
    pipeline = Pipeline([('logistic_l1', logistic_model)])

    # SMOTE to handle imbalance
    X_train_model_arr = np.array(X_train_model)
    y_train_model_arr = np.array(y_train_model)
    smote = SMOTE(sampling_strategy=0.6, random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train_model_arr, y_train_model_arr)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train_model.columns)
    y_resampled = pd.Series(y_resampled)

    # Fit the model using the pipeline
    pipeline.fit(X_resampled, y_resampled)

    # Perform hyperparameter search using the pipeline
    best_params = logistic_model.search(X_resampled, y_resampled)
    print("Best hyperparameters:", best_params)

    # # Test model
    predictions = pipeline.predict(X_test_model)
    print("Predictions:", predictions)

    # Test the model using the test data
    logistic_model.test_logistic(X_test_model, y_test_model)

    path = os.path.join(cwd, os.getenv("LOGISTIC_PATH"))
    filename = "logisticregl1_model1.pkl"
    if os.path.isfile(os.path.join(path, filename)):
        timestamp = datetime.now().strftime("_backup_%Y%m%d")
        bkpfilename = filename.replace('.pkl', f'_{timestamp}.pkl')
        os.rename(os.path.join(path, filename),
                  os.path.join(path, bkpfilename))

    # file_name = os.getenv("LIGHTGBM_PATH")
    file_name = os.path.join(path, filename)

    # Save the model to pkl
    with open(file_name, 'wb') as model_file:
        pickle.dump(logistic_model.model, model_file)

    return logistic_model.model

# logistic_regression_model1 = run_logistic_regression_model("model1")
# logistic_regression_model2 = run_logistic_regression_model("model2")


# Function to get the data for light gbm
def get_data_for_light_gbm(df: pd.DataFrame, model_type: Literal["model1", "model2"]) -> pd.DataFrame:
    X_train_model, X_test_model, y_train_model, y_test_model = di.data_imputations(
        df, model_type)
    train_data_model = di.post_split_feature_extraction_training_data(
        X_train_model, y_train_model)
    test_data_model = di.post_split_feature_extraction_testing_data(
        X_test_model, y_test_model, model_type)

    train_data_model = train_data_model.dropna()
    test_data_model = test_data_model.dropna()

    # train_data_model = train_data_model.drop(columns= ['census_gender','census_ethnicity'])
    # test_data_model = test_data_model.drop(columns= ['census_gender','census_ethnicity'])

    # columns_bool = ['nccis_sensupport',
    #                'nccis_alternative_provision',
    #                'nccis_teenage_mother',
    #                'nccis_carer_not_own_child',
    #                'nccis_supervised_by_yots',
    #                'nccis_send',
    #                'nccis_caring_for_own_child',
    #                'nccis_care_leaver',
    #                'nccis_looked_after_in_care',
    #                'nccis_refugee_asylum_seeker',
    #                'nccis_pregnancy',
    #                'nccis_substance_misuse',
    #                'special_school_flag']
    #
    # cols_sep = ['excluded_ever_suspended',
    #            'excluded_ever_excluded',
    #            'excluded_exclusions_rescinded']
    #
    # train_data_model = train_data_model.drop(columns= columns_bool)
    # test_data_model = test_data_model.drop(columns= columns_bool)
    # train_data_model[cols_sep] = train_data_model[cols_sep].astype("category")
    # test_data_model[cols_sep] = test_data_model[cols_sep].astype("category")
    # drop_cols = ['index_of_multiple_deprivation_imd_score','employment_score_rate']
    # train_data_model = train_data_model.drop(columns= drop_cols)
    # test_data_model= test_data_model.drop(columns= drop_cols)

    # train_data_model.to_parquet("/home/workspace/files/intermediate/train_test/train_model.parquet")
    # test_data_model.to_parquet("/home/workspace/files/intermediate/train_test/test_model.parquet")
    # raise Exception
    X_train_model = train_data_model.drop(columns='nccis_status')
    y_train_model = train_data_model['nccis_status']
    X_test_model = test_data_model.drop(columns='nccis_status')
    y_test_model = test_data_model['nccis_status']

    return X_train_model, X_test_model, y_train_model, y_test_model


# Function to run Lightgbm
def run_lightgbm(df: pd.DataFrame, model_type: Literal["model1", "model2"]):
    X_train_model, X_test_model, y_train_model, y_test_model = get_data_for_light_gbm(
        df, model_type)
    # print(X_train_model.shape)
    # print(X_test_model.shape)
    # print(y_test_model.shape)
    # print(y_train_model.shape)

    X_train_stud_ids = pd.DataFrame(X_train_model['stud_id'])
    X_test_stud_ids = pd.DataFrame(X_test_model['stud_id'])

    X_train_model = X_train_model.drop(columns=['stud_id'])
    X_test_model = X_test_model.drop(columns=['stud_id'])

    # SMOTE to handle imbalance
    cat_vars = X_train_model.select_dtypes(include=['category']).columns
    cat_vars = cat_vars.values.tolist()

    smote = SMOTENC(categorical_features=cat_vars, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_model, y_train_model)

    # Create an instance of the LightGBM class
    lgbm = LightGBM()
    # Fit the model and get the trained model
    trained_model = lgbm.fit(X_resampled, y_resampled)
    print("Trained_Model")
    # Test the model using the trained model
    scores = lgbm.test_model_lightgbm(
        X_test_model, y_test_model, trained_model)
    print("Scores: ", scores)

    # Get the feature importances of the light gbm
    importance = trained_model.feature_importance()

    # Create a list of feature names
    feature_names = trained_model.feature_name()

    # Create a dictionary to store the feature names and importances
    feature_importance = dict(zip(feature_names, importance))

    # Sort the dictionary by importance score in descending order
    sorted_importance = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Print the feature names and their importances
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance}")

    # plot feature  importance
    plt.figure(figsize=(10, 6))
    lgbm.plot_importance(trained_model, max_num_features=20, height=0.5)
    plt.title("Feature Importance")
    plt.show()

    # Save the model to pkl

    path = os.path.join(cwd, os.getenv("LIGHTGBM_PATH"))
    filename = "lightgbm_model2.pkl"
    if os.path.isfile(os.path.join(path, filename)):
        timestamp = datetime.now().strftime("_backup_%Y%m%d")
        bkpfilename = filename.replace('.pkl', f'_{timestamp}.pkl')
        os.rename(os.path.join(path, filename),
                  os.path.join(path, bkpfilename))

    # file_name = os.getenv("LIGHTGBM_PATH")
    file_name = os.path.join(path, filename)
    with open(file_name, 'wb') as model_file:
        pickle.dump(trained_model, model_file)

    return scores

# Function to read the model and pre-process the file and then generate the output


def predictions_from_pkl(df: pd.DataFrame, model_type: Literal["model1", "model2"]) -> pd.DataFrame:
    if model_type == "model1":
        model_bytes = pkgutil.get_data(
            "neet.assets.models", "logisticregl1_model1.pkl")
        model = pickle.load(io.BytesIO(model_bytes))
        
        # Drop features not used during training
        trained_features = model.feature_names_in_
        df = df[trained_features]

        try:
            predictions = model.predict(df.to_numpy())
        except:
            predictions = model.predict(df)
            
        df['prediction'] = predictions

    elif model_type == "model2":
        model_bytes = pkgutil.get_data(
            "neet.assets.models", "lightgbm_model2.pkl")
        model = pickle.load(io.BytesIO(model_bytes))
        
        # Drop features not used during training
        trained_features = model.feature_name()
        df = df[trained_features]
        
        try:
            predictions = model.predict(df.to_numpy())
        except:
            predictions = model.predict(df)

    return predictions


def test_feature_extract_clean(df, model_type: str):
    if model_type == "model2":
        df = dcf.mapping_target_varible(df)
        df = dcf.encode_and_data_type_conversions(
            df,
            constants.COLUMNS_CATEGORICAL_NOMINAL_MODEL2,
            constants.COLUMNS_CATEGORICAL_ORDINAL_MODEL2,
            constants.COLUMNS_NUMERIC_MODEL2,
        )
        df["distance_from_home_to_school_km"] = df.apply(
            dcf.calculate_distance, axis=1)

        columns_to_drop = [
            "home_latitude",
            "home_longitude",
            "school_latitude",
            "school_longitude",
            "nccis_academic_age",
            "census_surname",
            "census_forename",
            "nccis_order_of_nccis_update",
            "census_cohort",
            "ks4_cohort",
            "nccis_confirmed_date",
            "census_estab",
            "ks4_estab",
            "excluded_year",
            "excluded_cohort",
            "school_postcode",
            "lsoa_name_2011",
            "postcode",
            "nccis_time_recorded",
            "nccis_parent",
            "ks4_pass_94",
        ]
        df = df.drop(columns=columns_to_drop)
        df = f.feature_extract_attendance(df)
        df = f.feature_extract_census(df)
        df = f.feature_extract_school_performance(df)
        df = f.feature_extract_ks4(df)
        df = df.dropna()
        df = df.drop(columns=["nccis_status"])        
        try:
            stud_ids = pd.DataFrame(df["stud_id"])
            df = df.drop(columns=['stud_id'])
        except:
            stud_ids = pd.DataFrame(df.index)

    elif model_type == "model1":
        df = dcf.mapping_target_varible(df)
        df = dcf.encode_and_data_type_conversions(df, constants.COLUMNS_CATEGORICAL_NOMINAL_MODEL1,
                                                  constants.COLUMNS_CATEGORICAL_ORDINAL_MODEL1,
                                                  constants.COLUMNS_NUMERIC_MODEL1)
        df['distance_from_home_to_school_km'] = df.apply(
            dcf.calculate_distance, axis=1)
        # This is temporary way
        columns_to_drop = ['home_latitude', 'home_longitude',
                           'school_latitude', 'school_longitude',
                           'september_guarantee_academic_age', 'census_surname', 'census_forename',
                           'september_guarantee_order_of_nccis_update', 'census_cohort', 'ks4_cohort',
                           'september_guarantee_confirmed_date', 'ks4_fsm', 'ks4_fsm6',
                           'census_estab', 'ks4_estab', 'excluded_year',
                           'excluded_cohort', 'school_postcode',
                           'lsoa_name_2011', 'postcode', 'september_guarantee_time_recorded',
                           'september_guarantee_parent', 'ks4_pass_94']
        df = df.drop(columns=columns_to_drop)
        df = f.feature_extract_attendance(df)
        df = f.feature_extract_census(df)
        df = f.feature_extract_school_performance(df)
        df = f.feature_extract_ks4(df)
        df = df.dropna()
        try:
            stud_ids = pd.DataFrame(df["stud_id"])
            df = df.drop(columns=['stud_id'])
        except:
            stud_ids = pd.DataFrame(df.index)
        df = f.normalize_the_numeric_val(df)
        df = df.drop(columns=["nccis_status"])
        columns_cat = df.select_dtypes(include=['category']).columns
        df = df.drop(columns=columns_cat)

    return stud_ids, df


# dataframe2 = preprocessor.preprocess_all_data(read_files(), "model2")
# dataframe2 = pd.read_parquet(os.getenv("INTERMEDIATE_PREPROC_MODEL2"),engine='pyarrow')
# dataframe1 = pd.read_parquet(os.getenv("INTERMEDIATE_PREPROC_MODEL1"),engine='pyarrow')
# print(dataframe1.shape)
# print(dataframe1.dtypes)
# dataframe1=dataframe1.head(1000)

#
# stud_ids1, dataframe1 = test_feature_extract_clean(dataframe1,"model1")
# print(dataframe1.columns)
#
#
# output1 = test_models_from_pkl(dataframe1, "model1")
# print(dataframe1.columns)
#
