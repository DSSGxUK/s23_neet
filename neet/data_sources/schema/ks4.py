from pandera import DataFrameSchema, Column, Check
from neet.data_sources.schema.dtypes import YesNoBool

_schema = DataFrameSchema(
    columns={
        "ks4_acadyr": Column(
            dtype="category",
            nullable=True,
            description=None,
            title=None,
        ),
        "stud_id": Column(
            dtype="str",
            checks=None,
            nullable=False,
            unique=False,
            description="Unique Identifier for each student",
            title="UID",
        ),
        "ks4_yeargrp": Column(
            dtype="Int8",
            checks=[
                Check.greater_than_or_equal_to(min_value=10.0),
                Check.less_than_or_equal_to(max_value=12.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "ks4_actyrgrp": Column(
            dtype="Int8",
            checks=[
                Check.greater_than_or_equal_to(min_value=11.0),
                Check.less_than_or_equal_to(max_value=11.0),
            ],
            nullable=True,
            regex=False,
            description=None,
            title=None,
        ),
        "ks4_la": Column(
            dtype="category",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "ks4_estab": Column(
            dtype="category",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "ks4_att8": Column(
            dtype=float,
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=90),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "ks4_pass_94": Column(
            dtype="Int8",
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=50),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "ks4_priorband_ptq_ee": Column(
            dtype="category",
            checks=[Check.isin(allowed_values=[1, 2, 3, 4])],
            nullable=True,
            description=None,
            title=None,
        ),    
    },
    coerce=True,
    strict="filter",
    name="ks4",
    ordered=False,
    report_duplicates="all",
    unique_column_names=True,
    add_missing_columns=False,
    title="Key stage 4 Attainment ",
    description="Attainment in GCSEs",
)
