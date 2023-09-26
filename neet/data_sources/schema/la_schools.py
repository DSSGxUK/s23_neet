from pandera import DataFrameSchema, Column, Check

schema = DataFrameSchema(
    columns={
        "estab": Column(
            dtype="Int64",
            nullable=False,
            unique=True,
            coerce=False,
            description=None,
            title=None,
        ),
        "perctot": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=3.7),
                Check.less_than_or_equal_to(max_value=20.2),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "ppersabs10": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=3.4),
                Check.less_than_or_equal_to(max_value=57.3),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "nor": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=57256.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "pnorg": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=100.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "pnorb": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=100.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "psenelse": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=100.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "psenelk": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=71.73913043478261),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "numeal": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=21586.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "pnumeal": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=98.1),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "pnumengfl": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=100.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "pnumuncfl": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=5.5),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "numfsm": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=14996.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "norfsmever": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=40.0),
                Check.less_than_or_equal_to(max_value=52231.0),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "pnumfsmever": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(min_value=1.941747572815534),
                Check.less_than_or_equal_to(max_value=82.5),
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "schname": Column(
            dtype=str,
            nullable=True,
            description=None,
            title=None,
        ),
        "postcode": Column(
            dtype="category",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "schooltype_y": Column(
            dtype="category",
            nullable=True,
            description=None,
            title=None,
        ),
        "issecondary": Column(
            dtype="boolean",
            nullable=True,
            description=None,
            title=None,
        ),
        "ispost16": Column(
            dtype="boolean",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
        "gender": Column(
            dtype="category",
            checks=[Check.isin(allowed_values=["Mixed", "Boys", "Girls"])],
            nullable=True,
            description=None,
            title=None,
        ),
        "ofstedrating": Column(
            dtype="category",
            checks=[
                Check.isin(
                    allowed_values=[
                        "Outstanding",
                        "Good",
                        "Requires improvement",
                        "Inadequate",
                        "Serious Weaknesses",
                        "Special Measures",
                    ]
                )
            ],
            nullable=True,
            description=None,
            title=None,
        ),
        "ofstedlastinsp": Column(
            dtype="object",
            checks=None,
            nullable=True,
            description=None,
            title=None,
        ),
    },
    coerce=True,
    strict="filter",
    name="la_schools",
    ordered=False,
    report_duplicates="all",
    unique_column_names=True,
    add_missing_columns=False,
    title="LA school data",
    description="Addtional data about each school in the local authority.",
)
