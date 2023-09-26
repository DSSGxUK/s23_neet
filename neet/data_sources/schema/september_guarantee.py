from pandas import Timestamp
from pandera import DataFrameSchema, Column, Check, Index, MultiIndex
from neet.data_sources.schema.dtypes import YesNoBool


_schema = DataFrameSchema(
    columns={
        "stud_id": Column(
            dtype=str,
            nullable=False,
            unique=False,
            required=True,
            description="Unique Identifier for each student",
            title="UID",
        ),
        "postcode": Column(
            dtype="category",
            # checks=[Check.not_equal_to("ZZ999ZZ"), Check.not_equal_to("ZZ99 9ZZ")],
            nullable=True,
            description=None,
            title=None,
        ),
    },
    coerce=True,
    strict="filter",
    name="september-guarantees",
    ordered=False,
    report_duplicates="all",
    unique_column_names=True,
    add_missing_columns=False,
    title="September Guarantees",
    description="Schema for Sept. Guarantees",
)
