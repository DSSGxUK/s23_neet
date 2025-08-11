from typing import Literal, List
import pkgutil
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from codes import activity_codes_higher_level

from neet.ml_logic.roni import calculate_roni_score
from neet.constants import (
    DATA_STREAMLIT_RAW_PATH,
    DATA_SYNTHETIC,
)

from neet.constants import DatasetType
from neet.interface.main import streamlit_predictions

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "showAxisDragHandles": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "editable": False,
}
# Define dict that contains every dataset as key-value-pairs.
DATASET_TYPES = {
    "attendance_y7": "Attendance (Year 7)",
    "attendance_y8": "Attendance (Year 8)",
    "attendance_y9": "Attendance (Year 9)",
    "attendance_y10": "Attendance (Year 10)",
    "attendance_y11": "Attendance (Year 11)",
    "census_y7": "Census (Year 7)",
    "census_y8": "Census (Year 8)",
    "census_y9": "Census (Year 9)",
    "census_y10": "Census (Year 10)",
    "census_y11": "Census (Year 11)",
    "exclusions_y11": "Exclusions (Year 11)",
    "ks4_y11": "KS4 Attainment",  # Always Y11
    "september-guarantee_y12": "September Gurantees (Year 12)",
    "nccis_mar_y12": "NCCIS (March Year 12)",
    "nccis_sep_y13": "NCCIS (Sept. Year 13)",
}

literal_dataset_types = Literal[
    "attendance_y7",
    "attendance_y8",
    "attendance_y9",
    "attendance_y10",
    "attendance_y11",
    "census_y7",
    "census_y8",
    "census_y9",
    "census_y10",
    "census_y11",
    "exclusions_y11",
    "ks4_y11",
    "september-guarantee_y12",
    "nccis_mar_y12",
    "nccis_sep_y13",
]

COHORTS = ["2018-19", "2019-20", "2020-21",
           "2021-22", "2022-23", "2023-24", "2024-25"]

literal_cohorts = Literal[
    "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]


@st.cache_data
def add_custom_css() -> None:
    """Outputs styling from CSS file on every page."""
    styles = """
    footer {
    display: none !important;
    }

h2 {
    margin-top: 2rem;
    }

section[data-testid='stSidebar'] ul {
    max-height: none;
    }

html {
    scroll-behavior: smooth;
    }

/* Add background color to metric container */
div[data-testid='metric-container'] {
    margin-bottom: 1rem;
    padding: 0.5rem 1.25rem;
    border-radius: 0.25rem;
    background-color: #F3EDF3;
    }
    """
    st.markdown("<style>" + str(styles, "utf-8") +
                "</style>", unsafe_allow_html=True)


def get_file_name(
    cohort: literal_cohorts,
    dataset_type: literal_dataset_types,
) -> str:
    """
    Creates the filename.
    Moved to a function in case we have adapt the naming.

    Args:
        dataset_type: The type of the dataset, e. g. NCCIS
        cohort: Cohort (year 11 based) of the dataset
    Return:
        filename: String of the filename
    """
    return cohort + "_" + dataset_type + ".csv"


def add_file_to_data_raw(
    df: pd.DataFrame, cohort: literal_cohorts, dataset_type: literal_dataset_types
) -> None:
    """
    Adds a file to the data_raw session state and saves it to disk. This contains a
    representation of all files in an dict of dicts. Files have
    to be validated by pandera before added to data_raw. Session state
    has to be available before this function is called.

    Dicts have the form: { dataset_type: str, cohort: str, data: pd.DataFrame }
    """
    key = cohort + "_" + dataset_type

    value_dict = {
        "dataset_type": dataset_type,
        "cohort": cohort,
        "data": df,
    }

    # Add data to the state
    st.session_state.data_raw[key] = value_dict

    # Save the file to disk
    fullpath = DATA_STREAMLIT_RAW_PATH / get_file_name(cohort, dataset_type)
    df.to_csv(fullpath, index=False)


def remove_file_from_data_raw(
    cohort: literal_cohorts, dataset_type: literal_dataset_types
) -> None:
    """
    Removes a file from the data_raw sessions state and the disk
    """

    # Delete sessions state
    key = cohort + "_" + dataset_type
    del st.session_state.data_raw[key]

    # Delete from disk
    fullpath = DATA_STREAMLIT_RAW_PATH / get_file_name(cohort, dataset_type)
    fullpath.unlink(missing_ok=True)


def reset_data_raw() -> None:
    st.session_state.data_raw = []


def initalize_global_state() -> None:
    """
    Initalizes all variables that have to be present in the session state
    for streamlit to work.
    """
    if "use_synthetic_data" not in st.session_state:
        st.session_state.use_synthetic_data = True
        set_data("synthetic")

    if "data_final" not in st.session_state:
        # Needs to be replaced with the pipeline and the result should be saved to disk
        st.session_state.data_final = []

    # Make data available for the following sessions states
    data = st.session_state.data_final

    # Get a list of all uids
    if "data_uids_facets" not in st.session_state:
        cache = data.reset_index()[["stud_id", "census_estab"]]
        cache["census_estab"] = cache["census_estab"].str.title()
        st.session_state.data_uids_facets = cache

    if "data_uids" not in st.session_state:
        st.session_state.data_uids = data.index.tolist()

    # Get a list of all estab ids
    if "data_final_estabs" not in st.session_state:
        # Each value should be unique.
        estabs = (st.session_state.data_final)["census_estab"].unique()
        estabs = pd.Series(estabs).dropna()
        st.session_state.data_final_estabs = {
            s.title() for s in estabs.unique()}

    if "data_roni_score" not in st.session_state:
        roni_scores = calculate_roni_score(data)
        st.session_state.data_roni_score = roni_scores

    # if "data_predictions" not in st.session_state:
    #    predictions = pd.read_csv()
    #    st.session_state.data_predictions = predictions

    # Check if Y12 cohorts is available
    if "y12_exists" not in st.session_state:
        st.session_state.y12_exists = True

    # Check if Y12 nccis data is available
    if "y12_nccis_exists" not in st.session_state:
        st.session_state.y12_nccis_exists = True

    # Check if Y13 cohorts is available
    if "y13_exists" not in st.session_state:
        st.session_state.y13_exists = False

    # Combines the lookups for Y11 and Y12 in one boolean
    if "y12_y13_exist" not in st.session_state:
        st.session_state.y12_y13_exist = (
            True
            if (st.session_state.y12_exists == True)
            and (st.session_state.y13_exists == True)
            else False
        )

    # Check if we only have y12 with no nccis data
    if "y12_no_nccis" not in st.session_state:
        if st.session_state.y12_exists and not st.session_state.y12_nccis_exists:
            st.session_state.y12_no_nccis = True
        else:
            st.session_state.y12_no_nccis = False


def initalize_data_raw_state() -> None:
    """
    Load the raw data from disk. Only used on the data upload page.
    Session state must be kept in-sync with file system.
    """

    if "data_raw" not in st.session_state:
        st.session_state.data_raw = {}

        uploads = DATA_STREAMLIT_RAW_PATH.glob("*.csv")

        # We should add a check for filenames here. Otherwise it will pick up any csv in the path.
        for file in uploads:
            file_name = file.stem
            cohort, dataset_type = file_name.split("_", 1)
            df = pd.read_csv(file)
            add_file_to_data_raw(df, cohort, dataset_type)

    if "file_schools_performance" not in st.session_state:
        st.session_state.file_schools_performance = None


def short_circuit_data() -> None:
    """
    If synthetic data is used show and information on top of the page.
    If no data is uploaded show an error message and stop the execution of
    other content on the page.
    """

    if st.session_state.use_synthetic_data == True:
        st.info("You are currently using synthetic data to preview the dashboard.")
        return

    # Is a pandas dataframe
    data = st.session_state.data_final

    if not isinstance(data, pd.DataFrame) or data.empty:
        st.error(
            "You have to upload data before you can use the dashboard. You can also use synthetic data to test the dashboard. Please check the instruction for more information."
        )
        st.stop()


def scroll_to_top() -> None:
    """Triggers JavaScript to scroll to the top of the page"""
    js = """
        <script>
            var body = window.parent.document.querySelector(".main");
            body.scrollTop = 0
        </script>"""
    components.html(js)


def map_activity_code_categories(series: pd.Series, mapping: dict) -> pd.Series:
    """
    Takes activity codes and replaced them with the category, e.g. Education or NEET.

    Args:
        series: series with activity codes
        mapping: maps activity codes to higher-level categories

    Returns:
        New series with the mapped data with categorical dtype
    """
    return series.map(mapping)


def set_data(source: Literal["synthetic", "model"]):
    """
    Sets and prepares the data for the dashboard.
    """

    # Read the correct data source
    if source == "model":
        data = None
        st.session_state.use_synthetic_data = False
    elif source == "synthetic":
        data = pd.read_csv(DATA_SYNTHETIC, low_memory=False)
        st.session_state.use_synthetic_data = True

    # Be sure that data is available
    if not isinstance(data, pd.DataFrame) or data.empty:
        del st.session_state.data_final
        return

    # Add additional dashboard columns
    data["dashboard_school_year"] = "Year 12"
    data["dashboard_activities"] = map_activity_code_categories(
        data["nccis_code"], activity_codes_higher_level
    )

    st.session_state.data_final = data.set_index("stud_id")


def set_synthetic_data():
    """
    If button is clicked boolean for synthetic data is changed
    """
    if st.session_state.use_synthetic_data == True:
        st.session_state.use_synthetic_data = False
    else:
        st.session_state.use_synthetic_data = True
        # Data should be read again.


def rearrange_data() -> None:
    """
    Rearranges from the dict data structure of the dashboard to
    structure of that is used by the pipeline. Gets data from the
    streamlit session state.
    """
    def search_data_raw_dict(data_raw: dict, target: str) -> List[pd.DataFrame]:
        """
        Searches the data_raw nested dict to find all pandas dfs for a certain
        dataset type. In the dashboard they are saved in the form "name_year" (e.g. attendance_y11),
        but for the modelling we need a list of dataframes per datasettype (e.g. on list
        for all five attendance dataframes)

        Args:
            data_raw: nested dict with following inner dict:
                {"dataset_type": dataset_type, "cohort": cohort  "data": pd.DataFrame}
            target: type of dataset to search, e.g. attendance

        Returns:
            list of dataframes for the target.
        """
        dfs = []

        for nested_dict in data_raw.values():
            id_value = nested_dict["dataset_type"].split("_")[0]
            if id_value == target:
                dfs.append(nested_dict["data"])
        return dfs

    school_performance = st.session_state.file_schools_performance
    data_raw = st.session_state.data_raw

    # Check that all files are available. Should not be necessary,
    # but never trust the frontend. Not a reliable check
    if not school_performance or len(data_raw) == 0:
        st.warning("Please upload data to calculate predictions.")
        st.stop()

    attendance = search_data_raw_dict(data_raw, "attendance")
    census = search_data_raw_dict(data_raw, "census")
    exclusions = search_data_raw_dict(data_raw, "exclusions")
    ks4 = search_data_raw_dict(data_raw, "ks4")
    september_guarantee = search_data_raw_dict(data_raw, "september-guarantee")
    nccis = search_data_raw_dict(data_raw, "nccis")
    # to list for pre-processing
    school_performance = [school_performance["df"]]

    return DatasetType(
        attendance,
        census,
        exclusions,
        ks4,
        nccis,
        september_guarantee,
        school_performance,
    )


def calculate_predictions() -> None:
    """
    Resort the datastructure to pass it to the pipeline.
    Results are saved to session state.
    """
    datasets = rearrange_data()

    with st.spinner("Calculating predictions ..."):
        scroll_to_top()
        # If we have no model three we can determine the correct model like that.
        if len(datasets.nccis) > 0:
            y_hat = streamlit_predictions(datasets, "model2")
            st.write(y_hat)
        else:
            y_hat = streamlit_predictions(datasets, "model1")
            st.write(y_hat)

    st.success("Predictions calculated")
    st.balloons()


def plotly_static(fig: go.Figure) -> go.Figure:
    """
    Helper function that makes a plotly chart static, but keeps the
    hover tooltips in place.

    Args:
        fig: A plotly figure object.

    returns:
        A plotly figures objects with added parameters.
    """
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(dragmode="pan", margin={"t": 40, "b": 40})
    return fig
