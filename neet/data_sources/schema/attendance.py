from pandera import DataFrameSchema, Column, Check

_schema = DataFrameSchema(
    columns={
        "stud_id": Column(
            dtype="string",
            checks=None,
            nullable=False,
            unique=False, # Schould be True
            required=True,
            description="Unique Identifier for each student",
            title="UID",
        ),
        "possible_sessions": Column(
            dtype="Int16",
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=800),
            ],
            nullable=True,
            description=None,
            title="Possible sessions",
        ),
        "attendance_count": Column(
            dtype="Int16",
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=800),
            ],
            nullable=True,
            unique=False,
            title="Attendance Count",
        ),
        "authorised_absence": Column(
            dtype="Int16",
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=800),
            ],
            nullable=True,
            description=None,
            title="Authorised Absence",
        ),
        "unauthorised_absence": Column(
            dtype="Int16",
            checks=[
                Check.greater_than_or_equal_to(min_value=0.0),
                Check.less_than_or_equal_to(max_value=800),
            ],
            nullable=True,
            description=None,
            title="Unauthorised Absence",
        ),
        "excluded_e_count": Column(
            dtype="Int16",
            checks=[
                Check.greater_than_or_equal_to(min_value=0),
                Check.less_than_or_equal_to(max_value=800),
            ],
            nullable=True,
            description=None,
            title="Excluded Count (E)",
        ),
    },
    coerce=True,
    strict="filter",
    name="attendance",
    report_duplicates="all",
    unique_column_names=True,
    add_missing_columns=False,
    title="Attendance",
    description="Schema for attendance data based on Department of Education Schemas.",
)
