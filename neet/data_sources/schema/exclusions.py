from pandera import DataFrameSchema, Column
from neet.data_sources.schema.dtypes import YesNoBool

_schema = DataFrameSchema(
    columns={
        "stud_id": Column(
            dtype="string",
            nullable=False,
            unique=False,
            description="Unique Identifier for each student",
            title="UID",
        ),
        "ever_suspended": Column(
            dtype=YesNoBool(),
            nullable=True,
            coerce=True,
            description=None,
            title=None,
        ),
        "ever_excluded": Column(
            dtype=YesNoBool(),
            nullable=True,
            coerce=True,
            description=None,
            title=None,
        ),
        "exclusions_rescinded": Column(
            dtype=YesNoBool(),
            nullable=True,
            coerce=True,
            description=None,
            title=None,
        ),
    },
    coerce=True,
    strict="filter",
    name="exclusions",
    ordered=False,
    report_duplicates="all",
    unique_column_names=True,
    add_missing_columns=False,
    title="Exclusions",
    description="Contains information about excluded students",
)
