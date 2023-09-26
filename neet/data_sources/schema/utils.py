from typing import Literal
import pandera as pa

from neet.data_sources.schema.attendance import _schema as attendance
from neet.data_sources.schema.census import _schema as census
from neet.data_sources.schema.exclusions import _schema as exclusions
from neet.data_sources.schema.ks4 import _schema as ks4
from neet.data_sources.schema.nccis import _schema as nccis
from neet.data_sources.schema.school_performance import _schema as school_performance
from neet.data_sources.schema.september_guarantee import _schema as sept_guarantee


def get_schema(
    dataset_type=Literal[
        "attendance",
        "census",
        "exclusions",
        "ks4",
        "nccis",
        "september-guarantee",
        "school-performance",
    ],
) -> pa.DataFrameSchema():
    """
    Get the correct schema from the yaml file. Should be used with
    pandera decorators for pipelines.

    Args:
        The dataset type as string is based on the differnt dataset
        types used by our tool

    Returns:
        pandera schema for the selected dataset type

    Raises:
        Value error if the provided dataset type is not one of the specified types.
    """

    match dataset_type:
        case "attendance":
            return attendance
        case "census":
            return census
        case "exclusions":
            return exclusions
        case "ks4":
            return ks4
        case "nccis":
            return nccis
        case "school-performance":
            return school_performance
        case "september-guarantee":
            return sept_guarantee
        case _:
            raise ValueError("Value passed for parameter dataset_type is not accepted.")
