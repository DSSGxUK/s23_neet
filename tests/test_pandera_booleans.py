import pytest
import numpy as np
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column
from neet.data_sources.schema import get_schema
from neet.data_sources.schema.dtypes import YesNoBool


@pytest.fixture
def schema_coerce_true():
    schema = DataFrameSchema(
        {
            "A": Column(YesNoBool(), coerce=True, nullable=True),
            "B": Column(YesNoBool(), coerce=True, nullable=True),
            "C": Column(YesNoBool(), coerce=True, nullable=True),
        }
    )
    return schema


@pytest.fixture(
    params=[
        {"A": ["yes", "Yes"], "B": ["No", "n"], "C": ["y", "N"]},
        {"A": [1, 0], "B": [0, 1], "C": [0, 1]},
        {"A": ["yes", "Yes"], "B": [np.NaN, "n"], "C": [pd.NA, "N"]},
        {"A": [True, False], "B": [0, 1], "C": ["Yes", "n"]},
    ]
)
def valid_df(request):
    return pd.DataFrame(request.param)


@pytest.fixture(
    params=[
        {"A": ["2", "1"], "B": ["value", "string"], "C": ["y", "N"]},
        {"A": [True, "Yes"], "B": ["Yes", "no"], "C": ["Y", "n"]},
        {"A": ["string", "1"], "B": ["True", "False"], "C": ["y", "N"]},
        {"A": [1, 2], "B": [0, 1], "C": [0, 1]},
        {"A": [1, True], "B": [0, 1], "C": [0, 1]},
    ]
)
def invalid_df(request):
    return pd.DataFrame(request.param)


def test_schema_boolean_valid(valid_df, schema_coerce_true):
    schema_coerce_true.validate(valid_df)


def test_schema_boolean_invalid(invalid_df, schema_coerce_true):
    with pytest.raises(pa.errors.SchemaError):
        schema_coerce_true.validate(invalid_df)
