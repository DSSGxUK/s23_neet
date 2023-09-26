import utils as ut
import pandas as pd
import streamlit as st
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


def calc_counts(series: pd.Series, mapping: dict = None, normalize: bool = True):
    """Helper functions that calculates normalized value counts with mapping"""
    count = (
        series.value_counts(normalize=normalize).sort_values(axis="index").reset_index()
    )

    if mapping:
        count[count.columns[0]] = count[count.columns[0]].map(mapping)  #

    return count


def figure_languages(
    selected: pd.DataFrame,
    total: pd.DataFrame | None = None,
    mapping: dict | None = None,
    faceted: bool = False,
) -> go.Figure:
    """
    Creates bar plot of languages

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared

    Returns:
        fig: plotly figure object
    """
    
    
    counts = selected["census_language"].value_counts()
    top = counts.nlargest(9).index.tolist()

    # This changes the dataframe which should not happen.
    selected.loc[~selected["census_language"].isin(top), "census_language"] = "Other"
    count_selected = calc_counts(
        selected["census_language"], mapping=None, normalize=False
    )

    # Check if activity is selected
    if faceted:
        total.loc[~total["census_language"].isin(top), "census_language"] = "Other"
        count_total = calc_counts(
            total["census_language"], mapping=None, normalize=False
        )

        count_selected["Compare"] = "Selected"
        count_total["Compare"] = "Cohort"

        combined = pd.concat([count_selected, count_total], axis=0)
        fig = px.bar(
            combined,
            x="census_language",
            y="count",
            color="Compare",
            labels={"census_language": "Language"},
            text_auto=True,
        )
    else:
        fig = px.bar(
            count_selected,
            x="census_language",
            y="count",
            labels={"census_language": "Language"},
            text_auto=True,
        )

    fig.update_yaxes(title="Count")
    fig.update_layout(
        barmode="group",
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )
    fig.update_traces(textposition="auto", cliponaxis=False)

    return fig


def figure_fsm(
    selected: pd.DataFrame, total: pd.DataFrame | None = None, faceted: bool = False
) -> go.Figure:
    count_selected = calc_counts(selected["census_fsme_on_census_day"])
    """
    Creates relative horizontal stacked bar plot free school meals

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared

    Returns:
        fig: plotly figure object
    """
    colors = [
        "#003E7C",
        "#134d8b",
        "#2f6BA8",
        "#4A8AC5",
        "#66ABE2",
        "#83CDFF",
    ]
    order = {
        "census_fsme_on_census_day": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
        ]
    }

    # Check if activity is selected
    if faceted:
        count_total = calc_counts(total["census_fsme_on_census_day"])
        count_selected["Compare"] = "Selected"
        count_total["Compare"] = "Cohort"

        combined = pd.concat([count_selected, count_total], axis=0)
        combined["census_fsme_on_census_day"] = combined[
            "census_fsme_on_census_day"
        ].astype(str)
        fig = px.bar(
            combined,
            x="proportion",
            y="Compare",
            color="census_fsme_on_census_day",
            orientation="h",
            text_auto=True,
            color_discrete_sequence=colors,
            category_orders=order,
        )
        fig.update_traces(opacity=0.5, selector=({"name": "Selected"}))

    else:
        # Add dummy to plotly works
        count_selected["Compare"] = "Cohort"
        count_selected["census_fsme_on_census_day"] = count_selected[
            "census_fsme_on_census_day"
        ].astype(str)

        fig = px.bar(
            count_selected,
            x="proportion",
            color="census_fsme_on_census_day",
            y="Compare",
            orientation="h",
            text_auto=True,
            color_discrete_sequence=colors,
            category_orders=order,
        )
        fig.update_yaxes(visible=False)

    fig.update_xaxes(
        title="Years with FSM on census day between Year 7 and Year 11 (max is 5)"
    )
    fig.update_layout(
        xaxis_tickformat=".0%",
        height=250,
        uniformtext_minsize=16,
        uniformtext_mode="hide",
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )

    return ut.plotly_static(fig)


def figure_attainment_att8(
    selected: pd.DataFrame, total: pd.DataFrame | None = None, faceted: bool = False
) -> go.Figure:
    fig = go.Figure()
    """
    Creates histpogram of attainment 8 years

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared

    Returns:
        fig: plotly figure object
    """

    # Check if activity is selected
    if faceted:
        fig.add_trace(
            go.Histogram(
                x=total["ks4_att8"],
                xbins={"start": 0, "size": 10},
                histnorm="percent",
                opacity=0.4,
                name="Full cohort",
                marker_color="rgb(131, 201, 255)",
            )
        )

    fig.add_trace(
        go.Histogram(
            x=selected["ks4_att8"],
            xbins={"start": 0, "size": 10},
            histnorm="percent",
            opacity=0.7,
            name="Selected",
            marker_color="rgb(0, 104, 201)",
        )
    )

    fig.update_yaxes(title="Distribution in percent")
    fig.update_xaxes(title="Key Stage 4 Attainment 8")
    fig.update_layout(
        barmode="overlay",
        yaxis_ticksuffix="%",
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )

    return ut.plotly_static(fig)


def figure_attainment_pass94(
    selected: pd.DataFrame, total: pd.DataFrame | None = None, faceted: bool = False
) -> go.Figure:
    fig = go.Figure()
    """
    Creates histogram for passes 94 in KS4

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared

    Returns:
        fig: plotly figure object
    """

    # Check if activity is selected
    if faceted:
        fig.add_trace(
            go.Histogram(
                x=total["ks4_pass_94"],
                xbins={"start": 0, "size": 2},
                histnorm="percent",
                opacity=0.4,
                name="Full cohort",
                marker_color="rgb(131, 201, 255)",
            )
        )

    fig.add_trace(
        go.Histogram(
            x=selected["ks4_pass_94"],
            xbins={"start": 0, "size": 2},
            histnorm="percent",
            opacity=0.7,
            name="Selected",
            marker_color="rgb(0, 104, 201)",
        )
    )

    fig.update_yaxes(title="Distribution in percent")
    fig.update_xaxes(title="Key Stage 4 Passes 9-4")
    fig.update_layout(
        barmode="overlay",
        yaxis_ticksuffix="%",
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )

    return ut.plotly_static(fig)


def figure_attainment_ks2(
    selected: pd.DataFrame,
    total: pd.DataFrame | None = None,
    mapping: dict | None = None,
    faceted: bool = False,
) -> go.Figure:
    """
    Creates horizontal stacked bar plot for attainment in KS2

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared

    Returns:
        fig: plotly figure object
    """
    count_selected = calc_counts(selected["ks4_priorband_ptq_ee"], mapping)

    # Check if activity is selected
    if faceted:
        count_total = calc_counts(total["ks4_priorband_ptq_ee"], mapping)
        count_selected["Compare"] = "Selected"
        count_total["Compare"] = "Cohort"

        combined = pd.concat([count_selected, count_total], axis=0)

        fig = px.bar(
            combined,
            x="proportion",
            y="Compare",
            color="ks4_priorband_ptq_ee",
            orientation="h",
            text_auto=True,
            color_discrete_map={
                "higher standard": "rgb(0, 104, 201)",
                "at expected standard": "rgb(131, 201, 255)",
                "below expected standard": "rgb(255, 43, 43)",
                "Not available": "lightgrey",
            },
            category_orders={
                "ks4_priorband_ptq_ee": [
                    "higher standard",
                    "at expected standard",
                    "below expected standard",
                    "Not available",
                ]
            },
        )
        fig.update_traces(opacity=0.5, selector=({"name": "Selected"}))

    else:
        # Add dummy to plotly works
        count_selected["groups"] = 0

        fig = px.bar(
            count_selected,
            x="proportion",
            y="groups",
            color="ks4_priorband_ptq_ee",
            orientation="h",
            text_auto=True,
            color_discrete_map={
                "higher standard": "rgb(0, 104, 201)",
                "at expected standard": "rgb(131, 201, 255)",
                "below expected standard": "rgb(255, 43, 43)",
                "Not available": "lightgrey",
            },
            category_orders={
                "ks4_priorband_ptq_ee": [
                    "higher standard",
                    "at expected standard",
                    "below expected standard",
                    "Not available",
                ]
            },
        )
        fig.update_yaxes(visible=False)

    fig.update_xaxes(title="Key Stage 2 Band")
    fig.update_layout(
        xaxis_tickformat=".0%",
        height=250,
        uniformtext_minsize=16,
        uniformtext_mode="hide",
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )

    return ut.plotly_static(fig)


def figure_attendance_unauthorised(
    selected: pd.DataFrame, total: pd.DataFrame | None = None, faceted: bool = False
) -> go.Figure:
    """
    Line chart of unauthorised attendance.

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        faceted: bool if selected and total should be compare
        
    Return:
        plotly figure
    """
    fig = go.Figure()

    years = ["Year 7", "Year 8", "Year 9", "Year 10", "Year 11"]

    unauthorised_selected = (
        selected[
            [
                "unauthorised_absence_7",
                "unauthorised_absence_8",
                "unauthorised_absence_9",
                "unauthorised_absence_10",
                "unauthorised_absence_11",
            ]
        ]
        .mean()
        .tolist()
    )
    fig.add_traces(go.Scatter(x=years, y=unauthorised_selected, name="Selected"))

    if faceted:
        unauthorised_total = (
            total[
                [
                    "unauthorised_absence_7",
                    "unauthorised_absence_8",
                    "unauthorised_absence_9",
                    "unauthorised_absence_10",
                    "unauthorised_absence_11",
                ]
            ]
            .mean()
            .tolist()
        )
        fig.add_traces(go.Scatter(x=years, y=unauthorised_total, name="Cohort"))

    fig.update_yaxes(rangemode="tozero", title="Sessions")
    fig.update_layout(
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )

    return ut.plotly_static(fig)


def figure_attendance_authorised(
    selected: pd.DataFrame, total: pd.DataFrame | None = None, faceted: bool = False
) -> go.Figure:
    """
    Line chart of authorised attendance.

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        faceted: bool if selected and total should be compare
        
    Return:
        plotly figure
    """

    fig = go.Figure()

    years = ["Year 7", "Year 8", "Year 9", "Year 10", "Year 11"]

    authorised_selected = (
        selected[
            [
                "authorised_absence_7",
                "authorised_absence_8",
                "authorised_absence_9",
                "authorised_absence_10",
                "authorised_absence_11",
            ]
        ]
        .mean()
        .tolist()
    )
    fig.add_traces(go.Scatter(x=years, y=authorised_selected, name="Selected"))

    if faceted:
        authorised_total = (
            total[
                [
                    "authorised_absence_7",
                    "authorised_absence_8",
                    "authorised_absence_9",
                    "authorised_absence_10",
                    "authorised_absence_11",
                ]
            ]
            .mean()
            .tolist()
        )
        fig.add_traces(go.Scatter(x=years, y=authorised_total, name="Cohort"))

    fig.update_yaxes(rangemode="tozero", title="Sessions")
    fig.update_layout(
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )

    return ut.plotly_static(fig)


def figure_sen_need(
    selected: pd.DataFrame,
    total: pd.DataFrame | None = None,
    mapping: dict | None = None,
    faceted: bool = False,
) -> go.Figure:
    """
    Creates horizontal stacked bar plot for attainment in KS2

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared
        
    Returns:
        fig: plotly figure object
    """
    count_selected = calc_counts(selected["census_senprovision_y11"], mapping)

    colors = ["#0068C9", "#3C95E4", "#83CDFF"]

    # Check if activity is selected
    if faceted:
        count_total = calc_counts(total["census_senprovision_y11"], mapping)
        count_selected["Compare"] = "Selected"
        count_total["Compare"] = "Cohort"

        combined = pd.concat([count_selected, count_total], axis=0)

        fig = px.bar(
            combined,
            x="proportion",
            y="Compare",
            color="census_senprovision_y11",
            orientation="h",
            text_auto=True,
            color_discrete_sequence=colors,
        )
        fig.update_traces(opacity=0.5, selector=({"name": "Selected"}))

    else:
        count_selected["groups"] = 0

        fig = px.bar(
            count_selected,
            x="proportion",
            y="groups",
            color="census_senprovision_y11",
            orientation="h",
            text_auto=True,
            color_discrete_sequence=colors,
        )
        fig.update_yaxes(visible=False)

    fig.update_xaxes(title="SEN need", range=[0, 1])
    fig.update_layout(
        xaxis_tickformat=".0%",
        height=250,
        legend={
            "title_text": "Years",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
        uniformtext_minsize=16,
        uniformtext_mode="hide",
    )

    return ut.plotly_static(fig)


def figure_sen_details(
    selected: pd.DataFrame,
    total: pd.DataFrame | None = None,
    mapping: dict | None = None,
    faceted: bool = False,
):
    """
    Figure combines primary and secondary SEN needs in a bar plot.
    
    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared
        
    Returns:
        fig: plotly figure object
    """
    selected = pd.concat(
        [selected["census_senneed1_y11"], selected["census_senneed2_y11"]]
    )
    # selected = selected.replace("ZERO", pd.NA)
    count_selected = calc_counts(selected, mapping, normalize=False)

    if faceted:
        total = pd.concat([total["census_senneed1_y11"], total["census_senneed2_y11"]])
        total = total.replace("ZERO", pd.NA)
        count_total = calc_counts(total, mapping, normalize=False)
        count_selected["Compare"] = "Selected"
        count_total["Compare"] = "Cohort"

        combined = pd.concat([count_selected, count_total], axis=0)

        fig = px.bar(
            combined,
            x="index",
            y="count",
            barmode="group",
            color="Compare",
            text_auto=True,
        )
    else:
        fig = px.bar(
            count_selected,
            x="index",
            y="count",
            text_auto=True,
        )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title="Count")
    fig.update_xaxes(title="", categoryorder="category ascending")
    fig.update_layout(
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
    )

    return ut.plotly_static(fig)


def figure_ofstedrating(
    selected: pd.DataFrame, total: pd.DataFrame | None = None, faceted: bool = False
) -> go.Figure:
    """
    Creates horizontal stacked bar plot for ofstedrating

    Args:
        selected: pd.DataFrame with selected subset of data
        data: pd.DataFrame with whole dataset
        faceted: bool if selected and total should be compared

    Returns:
        fig: plotly figure object
    """
    count_selected = calc_counts(selected["ofstedrating"])

    # Check if activity is selected
    if faceted:
        count_total = calc_counts(total["ofstedrating"])
        count_selected["Compare"] = "Selected"
        count_total["Compare"] = "Cohort"

        combined = pd.concat([count_selected, count_total], axis=0)

        fig = px.bar(
            combined,
            x="proportion",
            y="Compare",
            color="ofstedrating",
            orientation="h",
            text_auto=True,
            color_discrete_map={
                "Outstanding": "rgb(0, 104, 201)",
                "Good": "rgb(131, 201, 255)",
                "Requires improvement": "rgb(255, 171, 171)",
                "Serious Weaknesses": "rgb(255, 43, 43)",
            },
            category_orders={
                "ofstedrating": [
                    "Outstanding",
                    "Good",
                    "Requires improvement",
                    "Serious weakness",
                ]
            },
        )
        fig.update_traces(opacity=0.5, selector=({"name": "Selected"}))

    else:
        # Add dummy so plotly works
        count_selected["groups"] = 0

        fig = px.bar(
            count_selected,
            x="proportion",
            y="groups",
            color="ofstedrating",
            orientation="h",
            text_auto=True,
            color_discrete_map={
                "Outstanding": "rgb(0, 104, 201)",
                "Good": "rgb(131, 201, 255)",
                "Requires improvement": "rgb(255, 171, 171)",
                "Serious Weaknesses": "rgb(255, 43, 43)",
            },
            category_orders={
                "ofstedrating": [
                    "Outstanding",
                    "Good",
                    "Requires improvement",
                    "Serious Weaknesses",
                ]
            },
        )
        fig.update_yaxes(visible=False)

    fig.update_xaxes(title="Ofsted rating", range=[0, 1])
    fig.update_layout(
        xaxis_tickformat=".0%",
        height=250,
        legend={
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
        },
        uniformtext_minsize=16,
        uniformtext_mode="hide",
    )

    return ut.plotly_static(fig)


def figure_nccis_characteristics(
    selected: pd.DataFrame,
    total: pd.DataFrame | None = None,
    mapping: dict = None,
    faceted: bool = False,
) -> go.Figure:
    """
    Creates horizontal vertical bar plot for characteristics

    Args:
        selected: pd.DataFrame with selected subset of data
        total: pd.DataFrame with whole dataset.
        mapping: dict that maps values to human understandable values
        faceted: bool if selected and total should be compared
        
    Returns:
        fig: plotly figure object
    """

    counts = []
    for column in characteristics:
        count = selected[column].value_counts()
        # Only get the True value from counts
        count = count.loc[True] if len(count.index) > 1 else 0
        counts.append(count)

    fig = go.Figure()

    # Check if activity is selected
    if faceted:
        counts_total = []
        for column in characteristics:
            count_total = total[column].value_counts()
            # Only get the True value from counts
            count_total = count_total.loc[True] if len(count_total.index) > 1 else 0
            counts_total.append(count_total)

        fig.add_trace(
            go.Bar(name="Selected", x=list(characteristics.values()), y=counts)
        )
        fig.add_trace(
            go.Bar(name="Full Cohort", x=list(characteristics.values()), y=counts_total)
        )
    else:
        fig.add_trace(
            go.Bar(name="Selected", x=list(characteristics.values()), y=counts)
        )

    fig.update_yaxes(title="Count")
    fig.update_xaxes(title="Young persons characteristics")
    return ut.plotly_static(fig)


def figure_nccis_code_distribution(
    selected: pd.DataFrame,
    total: pd.DataFrame | None = None,
    mapping: dict = None,
    faceted: bool = False,
) -> go.Figure:
    """
    Creates horizontal vertical bar plot for activity codes

    Args:
        selected: pd.DataFrame with selected subset of data
        data: pd.DataFrame with whole dataset.
        mapping: dict that maps columns values to human readable names
        faceted: bool if selected and total should be compared

    Returns:
        fig: plotly figure object
    """
    selected_count = calc_counts(
        selected["nccis_code"], mapping=mapping, normalize=False
    )
    fig = px.bar(
        selected_count,
        x="count",
        y="nccis_code",
        color="nccis_code",
        # orientation="h",
        log_x=True,
        text_auto=True,
        color_discrete_map={
            "Full-time education - school sixth-form": "rgb(96, 70, 144)",
            "Full-time education - sixth-form college": "rgb(96, 70, 144)",
            "Full-time education - further education": "rgb(96, 70, 144)",
            "Full-time education - higher education": "rgb(96, 70, 144)",
            "Part-time education": "rgb(96, 70, 144)",
            "Gap year students": "rgb(96, 70, 144)",
            "Full-time education - other": "rgb(96, 70, 144)",
            "Special post-16 institution": "rgb(29, 105, 150)",
            "Full-time education :custodial institution (juvenile offender)": "rgb(29, 105, 150)",
            "Apprenticeship": "rgb(29, 105, 150)",
            "Full-time employment with study (regulated qualification)": "rgb(29, 105, 150)",
            "Employment without training": "rgb(29, 105, 150)",
            "Employment with training (other)": "rgb(29, 105, 150)",
            "Temporary employment": "rgb(29, 105, 150)",
            "Part-time employment": "rgb(29, 105, 150)",
            "Self-employment": "rgb(29, 105, 150)",
            "Self-employment with study (regulated qualification)": "rgb(29, 105, 150)",
            "Work not for reward with study (regulated qualification)": "rgb(29, 105, 150)",
            "ESFA funded work-based learning": "rgb(56, 166, 165)",
            "Other training": "rgb(56, 166, 165)",
            "DWP training and support programme": "rgb(56, 166, 165)",
            "Traineeship": "rgb(56, 166, 165)",
            "Supported Internship": "rgb(56, 166, 165)",
            "Re-engagement provision": "rgb(56, 166, 165)",
            "Working not for reward": "rgb(225, 124, 5)",
            "Not yet ready for work or learning": "rgb(225, 124, 5)",
            "Start date agreed (other)": "rgb(225, 124, 5)",
            "Start date agreed (RPA compliant)": "rgb(225, 124, 5)",
            "Seeking employment, education or training": "rgb(225, 124, 5)",
            "Not available to labour market/learning - carer": "rgb(225, 124, 5)",
            "Not available to labour market/learning - teenage parent": "rgb(225, 124, 5)",
            "Not available to labour market/learning - illness": "rgb(225, 124, 5)",
            "Not available to labour market/learning - pregnancy": "rgb(225, 124, 5)",
            "Not available to labour market/learning - religious grounds": "rgb(225, 124, 5)",
            "Not available to labour market/learning - unlikely ever to be economically active": "rgb(225, 124, 5)",
            "Not available to labour market/learning - other reason": "rgb(225, 124, 5)",
            "Custody (young adult offender)": "rgb(237, 173, 8)",
            "Current situation not known": "rgb(237, 173, 8)",
            "Cannot be contacted - no current address": "rgb(237, 173, 8)",
            "Refused to disclose activity": "rgb(237, 173, 8)",
        },
    )

    fig.update_yaxes(title="NCCIS Activity Code")
    fig.update_xaxes(title="Count (logarithmic scale)")
    fig.update_layout(
        height=selected_count["nccis_code"].nunique() * 50,
        uniformtext_minsize=16,
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    return ut.plotly_static(fig)
