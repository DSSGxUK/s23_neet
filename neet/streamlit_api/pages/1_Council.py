import streamlit as st
from pathlib import Path
import utils as ut
import pandas as pd
from pandas.testing import assert_frame_equal
import plotly.express as px
import plotly.graph_objects as go
from utils import PLOTLY_CONFIG
from codes import (
    activity_codes_higher_level,
    ks4_priorband_ptq_ee,
    senprovision,
    sentypes,
    activity_codes,
    characteristics,
)

from plots import (
    figure_languages,
    figure_fsm,
    figure_attainment_att8,
    figure_attainment_ks2,
    figure_attainment_pass94,
    figure_attendance_authorised,
    figure_attendance_unauthorised,
    figure_nccis_characteristics,
    figure_nccis_code_distribution,
    figure_ofstedrating,
    figure_sen_details,
    figure_sen_need,
)

st.set_page_config(
    page_title="Council - NEETalert",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get the possible higher level categories and keep the order (little hacky)
ACTIVITIES = list(dict.fromkeys(activity_codes_higher_level.values()))


def set_select_school_year_index() -> int:
    """
    If only Y12 or Y13 data is availabe the disabled checkbox should
    select the available dataset by default.

    Takes all data from the sessions state and sets and index for the
    following list ["Both", "Year 12, "Year 13"]

    Returns;
        index for streamlit selectbox.
    """
    y12 = st.session_state.y12_exists
    y13 = st.session_state.y13_exists

    if y12 == True and y13 == True:
        return 0
    elif y12 == True and y13 != True:
        return 1
    elif y12 != True and y13 == True:
        return 2
    # To avoid exception
    else:
        return 0


def set_facet_items() -> None:
    """
    Filters the data based on the selected facets.
    Does not manipulate data_final in session state because if would
    have unexpected consequences for other pages.
    """
    year = st.session_state.select_school_year

    # Reset activities if year 12 with no nccis data is selected
    if st.session_state.y12_no_nccis and year == "Year 12":
        st.session_state.select_activities = []

    activities = st.session_state.select_activities
    data = st.session_state.data_final

    options_year = ["Year 12", "Year 13"]

    # We always have to check for existence of activties, because it can return an empty list
    activs_condition = activities and set(activities).issubset(set(ACTIVITIES))

    if year in options_year and activs_condition:
        filtered = data[
            (data["dashboard_activities"].isin(activities))
            & (data["dashboard_school_year"] == year)
        ]
    elif activities and activs_condition:
        filtered = data[data["dashboard_activities"].isin(activities)]
    elif year in ["Year 12", "Year 13"]:
        filtered = data[data["dashboard_school_year"] == year]
    else:
        filtered = data

    st.session_state.data_council = filtered


def main():
    """Renders the contents of the streamlit page."""

    # Add global styles and load state
   
    ut.initalize_global_state()
    ut.short_circuit_data()

    # We do not want to filter data final, because that
    # will cause problems on page change and for comparing.
    if "data_council" not in st.session_state:
        st.session_state.data_council = st.session_state.data_final

    with st.sidebar:
        st.selectbox(
            "Select school year",
            options=["Both", "Year 12", "Year 13"],
            key="select_school_year",
            index=set_select_school_year_index(),
            disabled=(not st.session_state.y12_y13_exist),
            on_change=set_facet_items,
            help="Can only be used if data for two cohort is available",
        )
        st.multiselect(
            "Select most recent activities",
            options=ACTIVITIES,
            help="Can only be used if activity data is availabe",
            key="select_activities",
            disabled=(
                st.session_state.y12_no_nccis
                and (
                    not st.session_state.y13_exists
                    or st.session_state.select_school_year == "Year 12"
                )
            ),
            on_change=set_facet_items,
        )

    st.title("NEETalert Overview", anchor=False)

    # To reset filtered data on page change.
    if len(st.session_state.select_activities) == 0:
        set_facet_items()

    data = st.session_state.data_final
    selected = st.session_state.data_council
    faceted = True if len(st.session_state.select_activities) > 0 else False

    # Data should be available
    if selected.empty:
        st.warning("No data available for your selections.")
        st.stop()

    st.markdown(f"*Selected {len(selected.index)} out of {len(data.index)} students*")

    st.header("Education during KS3 and KS4")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Languages")
        st.plotly_chart(
            figure_languages(selected, data, faceted),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with col2:
        st.subheader("Free school meals")
        st.plotly_chart(
            figure_fsm(selected, data, faceted),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    st.subheader("Attainment")

    st.markdown(
        f"{(selected['ks4_att8'] == 0).sum()} out of {selected['ks4_att8'].count()} students did not pass any GCSE."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            figure_attainment_att8(selected, data, faceted),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with col2:
        st.plotly_chart(
            figure_attainment_pass94(selected, data, faceted),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    st.plotly_chart(
        figure_attainment_ks2(selected, data, ks4_priorband_ptq_ee, faceted),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.subheader("Attendance")
    (
        col1,
        col2,
    ) = st.columns(2)

    with col1:
        st.markdown("**Authorised**")
        st.plotly_chart(
            figure_attendance_authorised(selected, data, faceted),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with col2:
        st.markdown("**Unauthorised**")
        st.plotly_chart(
            figure_attendance_unauthorised(selected, data, faceted),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    st.subheader("Special education needs in Year 11")
    st.plotly_chart(
        figure_sen_need(selected, data, senprovision, faceted),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.markdown("Combined primary and secondary SEN needs:")
    st.plotly_chart(
        figure_sen_details(selected, data, sentypes, faceted),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.subheader("Ofsted rating")
    st.plotly_chart(
        figure_ofstedrating(selected, data, faceted),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.header("Post-16 activies")
    st.subheader("Count of characteristics")
    st.plotly_chart(
        figure_nccis_characteristics(selected, data, characteristics, faceted),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.subheader("Count of activities")
    st.plotly_chart(
        figure_nccis_code_distribution(selected, data, activity_codes, faceted),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


# Run the Streamlit app
if __name__ == "__main__":
    main()
