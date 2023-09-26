from pandera import DataFrameSchema, Column, Check
from neet.data_sources.schema.dtypes import YesNoBool

"""
Missing Checks:
Ethnicity 
Language
"""

_schema = DataFrameSchema(
    columns={
        "stud_id": Column(
            dtype="string",
            checks=None,
            nullable=False,
            unique=False,
            description="Unique Identifier for each student",
            title="UID",
        ),
        "date_of_birth": Column(
            dtype="string",
            checks=None,
            nullable=True,
            description="For our model we are only using the month of birth",
            title="Date of Birth",
        ),
        "forename": Column(
            dtype="string",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "surname": Column(
            dtype="string",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "estab": Column(
            dtype="category",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "gender": Column(
            dtype="category",
            checks=[Check.isin(allowed_values=["M", "F", "m", "f"])],
            nullable=True,
            description=None,
            title=None,
        ),
        "entry_date": Column(
            dtype="object",
            nullable=True,
            description=None,
            title=None,
        ),
        "ncyear_actual": Column(
            dtype="Int8",
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=14),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "ethnicity": Column(
            dtype=str,
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "language": Column(
            dtype=str,
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "senprovision": Column(
            dtype="string",
            checks=[Check.isin(allowed_values=["S", "N", "K", "E"])],
            nullable=True,
            description=None,
            title=None,
        ),
        "senneed1": Column(
            dtype="category",
            checks=[
                Check.isin(
                    allowed_values=[
                        0,
                        "SPLD",
                        "MLD",
                        "SLD",
                        "PMLD",
                        "SLCN",
                        "HI",
                        "VI",
                        "MSI",
                        "PD",
                        "ASD",
                        "OTH",
                        "SEMH",
                        "NSA",
                    ]
                )
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "senneed2": Column(
            dtype="category",
             checks=[
                Check.isin(
                    allowed_values=[
                        0,
                        "SPLD",
                        "MLD",
                        "SLD",
                        "PMLD",
                        "SLCN",
                        "HI",
                        "VI",
                        "MSI",
                        "PD",
                        "ASD",
                        "OTH",
                        "SEMH",
                        "NSA",
                    ]
                )
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "senunit_indicator": Column(
            dtype=YesNoBool(),
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "resourced_provision_indicator": Column(
            dtype=YesNoBool(),
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "fsme_on_census_day": Column(
            dtype=YesNoBool(),
            coerce=True,
            nullable=True,
            description=None,
            title=None,
        ),
        "age": Column(
            dtype="Int8",
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=18),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
    },
    coerce=True,
    strict="filter",
    name="census",
    ordered=False,
    report_duplicates="all",
    unique_column_names=True,
    add_missing_columns=False,
    title="Census Data",
    description="Schema for Census data based on Department of Education Schema.",
)
