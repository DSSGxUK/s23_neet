import pytest
import numpy as np
import pandas as pd

from neet.data_sources.preprocessing_functions import compare_year_actual_to_year


@pytest.fixture(
    params=[
        {
            "stud_id": ["1", "2", "3", "4", "5"],
            "ncyear_actual": [12, 12, 10, np.NaN, np.NaN],
            "year": [12, 12, np.NaN, 12, np.NaN],
        },
    ]
)
def valid_df(request):
    return pd.DataFrame(request.param)


@pytest.fixture(
    params=[
        {"stud_id": ["1", "2"], "ncyear_actual": [12, 12], "year": [12, 13]},
        {"stud_id": ["1", "2"], "ncyear_actual": [12, 11], "year": [12, 12]},
        {
            "stud_id": ["1", "2", "2", "2"],
            "ncyear_actual": [12, 7, 8, 9],
            "year": [12, np.NaN, 8, 10],
        },
        {
            "stud_id": ["1", "2", "2", "2", "3", "3"],
            "ncyear_actual": [12, 7, 8, 9, 7, 8],
            "year": [12, np.NaN, 8, 10, 8, 9],
        },
    ]
)
def invalid_df(request):
    return pd.DataFrame(request.param)


@pytest.fixture()
def compare_df():
    return pd.DataFrame({"stud_id": ["1"], "ncyear_actual": [12], "year": [12]})


def test_compare_year_actual_to_year_valid(valid_df):
    df = compare_year_actual_to_year(valid_df)
    assert df.shape == valid_df.shape


def test_compare_year_actual_to_year_invalid(invalid_df, compare_df):
    df = compare_year_actual_to_year(invalid_df)
    # Input should not be equal to output
    assert df.shape != invalid_df.shape
    # But should be equal to the compare_df
    assert df.shape == compare_df.shape
