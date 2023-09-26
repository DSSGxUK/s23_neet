import numpy as np
import pandas as pd
import pytest
from neet.data_sources.schema import get_schema


@pytest.fixture
def valid_value():
    return "attendance"


@pytest.fixture(params=[999, "invalid", True])
def invalid_value(request):
    return request.param


def test_get_schema_valid(valid_value):
    get_schema(valid_value)


def test_get_schema_invalid(invalid_value):
    with pytest.raises(
        ValueError, match="Value passed for parameter dataset_type is not accepted"
    ):
        get_schema(invalid_value)
