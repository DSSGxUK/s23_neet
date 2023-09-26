import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import utils as ut
from utils import PLOTLY_CONFIG
from codes import (
    activity_codes,
    ethnicity,
    gender,
    ks4_priorband_ptq_ee,
    binary,
    senprovision,
    sentypes,
    characteristics,
)


st.set_page_config(
    page_title="Indivdiuals - NEETalert",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded",
)


def get_selected_uid() -> str | None:
    """
    Helper function, because state can have the value 'Select'.
    To make it failsafe the state gets checked against set of all uids.

    Return:
        uid or None, if it is no valid uid.
    """
    uid = st.session_state.selected_uid
    uids = st.session_state.data_uids_facets

    if uid in uids["stud_id"].to_list():
        return uid

    return None


def set_facet_items() -> None:
    """Sets the values off facets based on the selected facets"""
    cache = st.session_state.data_uids_facets
    uid = st.session_state.selected_uid
    estab = st.session_state.selected_estab
    risk = st.session_state.slider_risk

    # Change uids cache df
    cache = cache[cache["census_estab"] == estab.title()]

    st.session_state.data_uids = cache["stud_id"].tolist()


def get_academic_year(series: pd.Series) -> int | None:
    """
    Determine the academic year of the young person

    Args:
        series: pd.Series with information about the student.

    Return:
        Academic year of the student (Y12 or Y13).
    """

    if series["nccis_academic_age"] == 16:
        return 12
    elif series["nccis_academic_age"] == 17:
        return 13

    return None


def figure_risk(series: pd.Series) -> go.Figure:
    """
    Figure for the risk of becoming NEET.

    Args:
        series: pandas series with values of the individual

    Return:
        plotly figure
    """

    def replace_values(list):
        """Replaces 0 and 1 with non-NEET and NEET string"""
        replacements = {0: "not-NEET", 1: "NEET"}

        for i, v in enumerate(list):
            if v in replacements:
                list[i] = replacements[v]

        return list

    activiy = [0, 1]
    activiy_codes = [activity_codes[key] for key in [210, 220]]
    riskscore = [None, None, "20%"]
    predicted = [None, None, 1]
    predicted_line = [None, 1, 1, None]

    for l in [activiy, predicted, predicted_line]:
        l = replace_values(l)

    years = [
        "Year 12 Sept",
        "Year 12 March",
        "Year 13 Sept",
        "Year 13 March",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=predicted_line,
            mode="lines",
            line={"dash": "dash", "color": "rgba(0, 104, 201, 0.5)"},
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="lines+markers",
            x=years,
            y=activiy,
            line={"dash": "solid", "color": "rgba(0, 104, 201, 1)"},
            marker={"size": 20},
            customdata=activiy_codes,
            hovertemplate="<b>Activity:</b>  <br> %{customdata}",
            hoverlabel={"font_size": 16},
            name="",  # removes name from hover
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=predicted,
            mode="markers",
            marker_symbol="circle",
            marker_line_width=5,
            marker_line_color="rgba(0, 104, 201, 0.5)",
            marker={"size": 17, "color": "rgba(255, 255, 255, 1)"},
            customdata=riskscore,
            hovertemplate="<b>Prediction of NEET status</b>  <br> Risk Score: %{customdata}",
            hoverlabel={"font_size": 16},
            name="",  # removes name from hover
        )
    )
    fig.update_yaxes(
        type="category",
        categoryorder="total ascending",
        title="NEET status/prediction",
        tickfont={"size": 20},
    )
    fig.update_xaxes(
        tickfont={"size": 15},
    )
    fig.update_layout(
        showlegend=False, dragmode="pan", yaxis={"fixedrange": True}, height=260
    )

    return ut.plotly_static(fig)


def figure_explainability(series: pd.Series):
    """
    Figure to explain the top factors for the individual prediction

    Args:
        series: pandas series with values of the individual

    Return:
        plotly figure
    """

    df = pd.DataFrame(
        {
            "values": [8, -4, 10, -1],
            "predictors": ["Attainment", "SEN", "Parent", "Attendance"],
        }
    )

    # Sort by absolute distance from zero
    df = df.sort_values(by="values", key=abs)

    # Add column with color
    df["color"] = np.where(
        df["values"] > 0, "rgba(0, 104, 201, 0.5)", "rgba(255, 43, 43, 0.5)"
    )
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["values"],
            y=df["predictors"],
            orientation="h",
            marker_color=df["color"],
            text=df["predictors"],
            textposition="auto",
            textfont={"size": 16},
            hoverinfo="none",
        )
    )

    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(title="Importance of factors on predicted risk score")
    fig.update_layout(height=300)

    return ut.plotly_static(fig)


def figure_attendance(series: pd.Series, cohort_data: pd.DataFrame) -> go.Figure():
    """
    Plot of authorised and unauthorised attendance.

    Args:
        series: pandas series with values of the individual
        cohort_data: pd.DataFrame with data for the whole cohort of selected
            young person. So young person's value can be compared.

    Return:
        plotly figure
    """

    years = [7, 8, 9, 10, 11]

    authorised = series.loc[
        [
            "authorised_absence_7",
            "authorised_absence_8",
            "authorised_absence_9",
            "authorised_absence_10",
            "authorised_absence_11",
        ]
    ].to_numpy()

    mean_authorised = (
        cohort_data[
            [
                "authorised_absence_7",
                "authorised_absence_8",
                "authorised_absence_9",
                "authorised_absence_10",
                "authorised_absence_11",
            ]
        ]
        .mean()
        .to_numpy()
    )

    sd_authorised = (
        cohort_data[
            [
                "authorised_absence_7",
                "authorised_absence_8",
                "authorised_absence_9",
                "authorised_absence_10",
                "authorised_absence_11",
            ]
        ]
        .std()
        .to_numpy()
    )

    unauthorised = series.loc[
        [
            "unauthorised_absence_7",
            "unauthorised_absence_8",
            "unauthorised_absence_9",
            "unauthorised_absence_10",
            "unauthorised_absence_11",
        ]
    ].to_numpy()

    mean_unauthorised = (
        cohort_data[
            [
                "unauthorised_absence_7",
                "unauthorised_absence_8",
                "unauthorised_absence_9",
                "unauthorised_absence_10",
                "unauthorised_absence_11",
            ]
        ]
        .mean()
        .to_numpy()
    )

    sd_unauthorised = (
        cohort_data[
            [
                "unauthorised_absence_7",
                "unauthorised_absence_8",
                "unauthorised_absence_9",
                "unauthorised_absence_10",
                "unauthorised_absence_11",
            ]
        ]
        .std()
        .to_numpy()
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=["Authorised Abscense", "Unauthorised Abscense"],
    )

    fig.add_trace(
        go.Scatter(
            x=years + years[::-1],
            y=list(mean_authorised + sd_authorised)
            + list((mean_authorised - sd_authorised)[::-1]),
            fill="toself",
            fillcolor="rgba(0, 104, 201, 0.1)",
            line={"color": "rgba(0, 0, 0, 0)"},
            name="Cohort Deviation",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=authorised,
            mode="lines+markers",
            name="Individual",
            line={"color": "rgba(0, 104, 201, 1)"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=years + years[::-1],
            y=list(mean_unauthorised + sd_unauthorised)
            + list((mean_unauthorised - sd_unauthorised)[::-1]),
            fill="toself",
            fillcolor="rgba(255, 43, 43, 0.1)",
            line={"color": "rgba(0, 0, 0, 0)"},
            name="Cohort Deviation",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=unauthorised,
            mode="lines+markers",
            name="Invidivual",
            line={"color": "rgba(255, 43, 43, 1)"},
        ),
        row=1,
        col=2,
    )

    # Calculate max y-axis value. Should be max value of authorised vs unauthorised
    # and either the area or the young persons value.
    max = np.nanmax(
        [
            authorised,
            unauthorised,
            (mean_unauthorised + sd_unauthorised),
            (mean_authorised + sd_authorised),
        ]
    )

    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Number of sessions", range=[0, max + 5])
    fig.update_layout(
        modebar_remove=["zoom"],
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.5, "xanchor": "left"},
    )

    return ut.plotly_static(fig)


def figure_attainment_att8(
    series: pd.Series, cohort_data: pd.DataFrame, name: str
) -> go.Figure:
    """
    Plot of Attainment 8, one of our two KS4 predictors.

    Args:
        series: data for the young person.
        cohort_data: data for the whole cohort of the young person.
        name: Name of the young person

    Returns:
        plotly figure
    """

    fig = go.Figure()

    fig.add_trace(
        go.Violin(
            x=cohort_data["ks4_att8"],
            hoverinfo="skip",
            fillcolor="rgba(0, 104, 201, 0.15)",
            meanline_visible=True,
            side="positive",
            name="",
        ),
    )
    fig.add_vline(
        x=series["ks4_att8"],
        annotation={"font_size": 16},
        annotation_text=name + f" <br>(Score {series['ks4_att8']})",
        annotation_position="top",
    )

    fig.update_xaxes(range=[0, cohort_data["ks4_att8"].max()])
    fig.update_layout(showlegend=False, height=180, margin={"l": 80, "r": 80})

    return ut.plotly_static(fig)


def figure_attainment_pass_94(
    series: pd.Series, cohort_data: pd.DataFrame, name: str
) -> None:
    """
    Plot of KS4 Passes 94.

    Args:
        series: data for the young person.
        cohort_data: data for the whole cohort of the young person.
        name: Name of the young person

    Returns:
        plotly figure
    """

    fig = go.Figure()

    fig.add_trace(
        go.Violin(
            x=cohort_data["ks4_pass_94"],
            hoverinfo="skip",
            fillcolor="rgba(0, 104, 201, 0.15)",
            meanline_visible=True,
            side="positive",
            name="",
        ),
    )
    fig.add_vline(
        x=series["ks4_pass_94"],
        annotation={"font_size": 16},
        annotation_text=name + f"<br> (Score {series['ks4_pass_94']})",
        annotation_position="top",
    )

    fig.update_xaxes(range=[0, cohort_data["ks4_pass_94"].max()])
    fig.update_layout(showlegend=False, height=200, margin={"l": 80, "r": 80})

    return ut.plotly_static(fig)


def main():
    """Renders the contents of the streamlit page."""

    # Add global styles and load state

    ut.initalize_global_state()
    ut.short_circuit_data()

    # Get the data from state
    data = st.session_state.data_final
    uids_cache = st.session_state.data_uids_facets
    uids = st.session_state.data_uids

    with st.sidebar:
        st.selectbox(
            "Filter by KS4 establishment",
            options=np.insert(np.unique(uids_cache["census_estab"]), 0, "Select"),
            key="selected_estab",
            on_change=set_facet_items,
        )
        st.slider(
            "Filter by predicted risk (percent)",
            min_value=0,
            max_value=100,
            step=1,
            value=[0, 100],
            key="slider_risk",
            on_change=set_facet_items,
        )
        st.selectbox(
            "Select a young person",
            options=["Select"] + uids,
            key="selected_uid",
            format_func=(
                lambda x: "Select" if x == "Select" else "Name (" + str(x) + ")"
            ),  # + data.at[x, 'upn']
            help="You can type to search for a student",
        )

    uid = get_selected_uid()

    # Bail early if no uid is selected
    if uid is None:
        st.title("Information about one young person")
        st.warning("Please select a young person")
        st.stop()

    # TODO: Bail early if that student was dropped in between.
    # TODO: Check is student is model 1, 2, 3, or no model.

    series = data.loc[uid]

    cohort_data = data.loc[data["census_cohort"] == series["census_cohort"]]

    if "census_forename" in series.index and "census_surname" in series.index:
        name = str(series["census_forename"] + " " + series["census_surname"])
        st.title(f"About {name}")
    else:
        name = "Selected young person"
        st.title("About the young person")

    st.markdown(
        f"<span style='font-size:large'>Currently in **Year {get_academic_year(series)}** (Cohort {series['census_cohort']})</span>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Risk of NEET",
        value="76%",
        delta_color="off",
        help="Risk score calculated by out prediction model.",
    )

    #roni_score = (st.session_state.data_roni_score).loc[uid, "roni_score"]
    roni_score = 4
    
    col2.metric(
        "RONI Score",
        value=roni_score,
        help="To learn more about our RONI score calculation, please read the desciption on the'About' page.",
    )

    attendance_percentage = (
        (series["unauthorised_absence_11"] + series["authorised_absence_11"])
        / series["possible_sessions_11"]
        * 100
    )

    col3.metric(
        "Percentage Absence (Year 11)",
        value=str(round(attendance_percentage, ndigits=1)) + "%",
    )

    col4.metric(
        "KS4 Attainment 8",
        value=series["ks4_att8"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""**UID:** {uid}<br> 
            **Ethnicity:** {ethnicity.get(series["census_ethnicity"])} <br>
            **Gender:** {gender.get(series["census_gender"])}<br> 
            **Language:** {series["census_language"]}<br> 
            **Postcode:** {series["postcode"]}""",
            unsafe_allow_html=True,
        )

    with col2:
        fsm = str(f"{series['census_fsme_on_census_day']} years (Y7 to Y11)")
        st.markdown(
            f"""**Establishment (KS4):** {series["census_estab"].title()}<br>
            **Enrolment Status (KS4):** {series["census_enrol_status"]}<br>
            **Free School Meals:** {fsm} """,
            unsafe_allow_html=True,
        )
        # Only print SEN information if flag is set.
        if series["census_senprovision_y11"] in ["K", "E"]:
            st.markdown(
                f"""**SEN Provision:** {senprovision.get(series["census_senprovision_y11"])}<br> 
                **Primary SEN need:** {sentypes.get(series["census_senneed1"])}<br> 
                **Secondary SEN need:** {sentypes.get(series["census_senneed2"])}""",
                unsafe_allow_html=True,
            )

    # Check for one column name for know if nccis is available
    nccis_available = True if "nccis_pregnancy" in series.index else False

    # Only show this information if nccis is available.
    if nccis_available:
        st.header("Post-16 activities (NCCIS)", anchor=False)
        info = series.loc[characteristics.keys()]
        info = info[info == "Y"]
        st.markdown(f'**Current activity:** {activity_codes.get(series["nccis_code"])}')
        if not info.empty:
            string = ", ".join(map(str, map(characteristics.get, info.index)))
            st.markdown(f"**Characteristics:** {string}")

        st.subheader("Area of living", anchor=False)
        st.markdown(
            f"""**Postcode:** {series["postcode"]}<br>
                **Index of Deprivation:** {series["index_of_multiple_deprivation_imd_score"]}<br> 
                **LSOA name:** {series["lsoa_name_2011"]}<br>""",
            unsafe_allow_html=True,
        )

    # We do not have this data. So lets not render it.
    st.header("Risk prediciton over time", anchor=False)
    st.markdown("Shows the NEET past status and the NEET prediction in 6 months.")
    st.plotly_chart(
        figure_risk(series),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    # st.error("We dont have this data so we cannot render this figure")

    st.header("Factors driving the risk score", anchor=False)
    st.plotly_chart(
        figure_explainability(series),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.header("Attendance", anchor=False)
    st.markdown(
        f"""**Young person suspended:** {binary.get(series["excluded_ever_suspended"])}<br> 
        **Young person excluded:** {binary.get(series["excluded_ever_excluded"])}<br> 
        **Exclusion rescinded:** {binary.get(series["excluded_exclusions_rescinded"])}<br> """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Figure shows attendance between Year 7 and Year 11. Shaded background shows the cohort distribution."
    )
    st.plotly_chart(
        figure_attendance(series, cohort_data),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.header("Academic attainment", anchor=False)
    st.markdown("Academic Attainment in GCSEs compared to the whole cohort.")
    st.markdown(
        f'Key stage 2 attainment of the young person was **{ks4_priorband_ptq_ee.get(series["ks4_priorband_ptq_ee"])}**'
    )
    st.subheader("Attainment 8", anchor=False)
    st.plotly_chart(
        figure_attainment_att8(series, cohort_data, name),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.subheader("Passes 9 to 4", anchor=False)
    st.plotly_chart(
        figure_attainment_pass_94(series, cohort_data, name),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


# Run the Streamlit app
if __name__ == "__main__":
    main()
