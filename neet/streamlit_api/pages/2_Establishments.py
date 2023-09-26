import numpy as np
import pandas as pd
import streamlit as st
import utils as ut

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
    page_title="Establishment - NEETalert",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_selected_estab() -> str | None:
    """
    Helper function, because state can have the value 'Select'.
    To make it failsafe the state gets checked against set of all estab ids.

    Return:
        uid or None, if it is no valid estab id.

    TODO: Might make sense to move set of uids into the state as part of initalize_global_state()
    """

    selected = st.session_state.selected_estab
    estabs = st.session_state.data_final_estabs

    if selected in estabs:
        return selected

    return None


def main():
    """Renders the contents of the streamlit page."""

    # Add global styles and load state
  
    ut.initalize_global_state()
    ut.short_circuit_data()

    # Get the data from state
    data = st.session_state.data_final
    all_estabs = st.session_state.data_final_estabs

    with st.sidebar:
        st.selectbox(
            "Select Establishments",
            options=["Select"] + sorted(all_estabs),
            key="selected_estab",
            format_func=(
                lambda x: "Select" if x == "Select" else str(x)
            ),  # + data.at[x, 'upn']
        )
        st.markdown("This list is based on the establishment a young person attended in Year 11.")

    estab = get_selected_estab()

    st.title("Information about establishments")

    # Bail early if no uid is selected
    if not estab:
        st.warning("Please select one establishment")
        return

    #estab_data = data[data["census_estab"].str.lower().isin(map(str.lower, estab))]
    selected = data.loc[data["census_estab"].str.lower() == estab.lower()]
    
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
            figure_languages(selected, data, faceted=True),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with col2:
        st.subheader("Free school meals")
        st.plotly_chart(
            figure_fsm(selected, data, faceted=True),
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
            figure_attainment_att8(selected, data, faceted=True),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with col2:
        st.plotly_chart(
            figure_attainment_pass94(selected, data, faceted=True),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    st.plotly_chart(
        figure_attainment_ks2(selected, data, ks4_priorband_ptq_ee, faceted=True),
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
            figure_attendance_authorised(selected, data, faceted=True),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with col2:
        st.markdown("**Unauthorised**")
        st.plotly_chart(
            figure_attendance_unauthorised(selected, data, faceted=True),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    st.subheader("Special education needs in Year 11")
    st.plotly_chart(
        figure_sen_need(selected, data, senprovision, faceted=True),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    st.markdown("Combined primary and secondary SEN needs:")
    st.plotly_chart(
        figure_sen_details(selected, data, sentypes, faceted=True),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.header("Post-16 activies")
    st.subheader("Count of characteristics")
    st.plotly_chart(
        figure_nccis_characteristics(selected, data, characteristics, faceted=True),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.subheader("Count of activities")
    st.plotly_chart(
        figure_nccis_code_distribution(selected, data, activity_codes, faceted=True),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    
    
# Run the Streamlit app
if __name__ == "__main__":
    main()
