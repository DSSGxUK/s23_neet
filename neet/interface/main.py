import numpy as np
import pandas as pd
from typing import Literal, Tuple
import neet.data_sources.preprocessing_functions as preproc
import neet.data_sources.constants as constants
from neet.data_sources.schema import get_schema
from neet.constants import DATA_DEVELOP_RAW_PATH, DatasetType
from neet.data_sources.local import read_datasets
import neet.ml_logic.preprocessor as preprocessor
import neet.ml_logic.model as model
import sys

def preprocess(model_type:str) -> pd.DataFrame:
    """
    function to pre process data and return the joined dataframe
    """

    #preprocess data from local
    df = preprocessor.preprocess_all_data(read_datasets(), model_type)
    
    print(df.shape)
    
    return df


def train(df, model_type):
    """
    Train a new model on the full (already pre-processed) dataset

    """

    if model_type == "model1":
        model.run_logistic_regression_model(df, model_type)
    elif model_type == "model2":
        model.run_lightgbm(df, model_type)

    return  None


def evaluate():
    """
    Evaluate the performance of the latest production model on new data

    """

    return  # here some metric


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    # Here our code #

    return  # y_pred


def package_name():
    print("The package name is neet!")


#######################
# Streamlit Interface #
#######################

def streamlit_predictions(
    datasets: DatasetType, model_type: Literal["model1", "model2", "model3"]
):
    """
    Calculates predictions and shapely based explanations for the dashboard.
    
    Args:
        datasets: namedtuple DatasetType
        model_type: the model to calculate. Depends on the available data.

    Returns:
        predictions: pd.DataFrame with modelling results.
        shap_values: np.array with shapely values for the predictions.
    """
    if model_type == "model3":
        raise Exception("This model does not exist")

    df = preprocessor.preprocess_all_data(datasets, model_type)

    uids, X = model.test_feature_extract_clean(df, model_type)

    y_hat = model.predictions_from_pkl(X, model_type)
    
    # Join uids and predictions again
    #predictions = y_hat

    return y_hat


if __name__ == "__main__":
    package_name()
    if len(sys.argv) < 2:
        print("Usage: python neet/interface/main.py model1")
        print("Usage: python neet/interface/main.py model2")
        sys.exit(1)
    model_type = sys.argv[1]
    print(f"Running model: {model_type}")
    
    df = preprocess(model_type)
    print(f"data has been preprocessed for {model_type}")
    
    train(df, model_type)
    print(f"{model_type} has been trained")

    
