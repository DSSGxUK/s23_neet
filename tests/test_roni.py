import numpy as np
import pandas as pd
import pytest
from neet.ml_logic.roni import calculate_roni_score


@pytest.fixture
def dataframe():
    columns = [
        "stud_id",
        "attendance_count_11",
        "possible_sessions_11",
        "census_language",
        "census_senprovision_y11",
        "excluded_ever_excluded",
        "census_fsme_on_census_day",
        "nccis_alternative_provision",
        "nccis_looked_after_in_care",
        "nccis_parent",
        "nccis_pregnancy",
        "nccis_carer_not_own_child",
        "nccis_supervised_by_yots",
        "nccis_code",
    ]

    data = [
        [1, 30, 300, "GER", "E", True, 5, True, True, True, False, True, True, 710],  # 9/20
        [2, 99, 100, "ENG", "M", False, 0, False, False, False, False, False, False, 110],  # 0/0
        [3, 88, 100, "GER", "K", False, 1, True, True, True, True, False, False, 540],  # 5/10
    ]

    return pd.DataFrame(data, columns=columns).set_index("stud_id")


def test_calculate_roni_score_without_nccis(dataframe):
    scores = calculate_roni_score(dataframe)

    scores_test = pd.DataFrame(
        data=[[1, 9, True], [2, 0, False], [3, 5, True]],
        columns=["stud_id", "roni_score", "roni_classification"],
    ).set_index("stud_id")

    pd.testing.assert_frame_equal(scores, scores_test)


def test_calculate_roni_score_with_nccis(dataframe):
    scores = calculate_roni_score(dataframe, nccis=True)

    scores_test = pd.DataFrame(
        data=[[1, 20, True], [2, 0, False], [3, 10, True]],
        columns=["stud_id", "roni_score", "roni_classification"],
    ).set_index("stud_id")

    pd.testing.assert_frame_equal(scores, scores_test)
