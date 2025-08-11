import sys, os

sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import utils as ut

st.set_page_config(
    page_title="Home - NEETalert",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded",
)


def main():
    """Renders the contents of the streamlit page."""

    # Add global styles and load state
    ut.initalize_global_state()

    data = st.session_state.data_final

    st.title("Welcome to NEETalert page! 👋", anchor=False)
    """
    **The project**

    This project aims to develop predictive models to identify individuals
    at risk of becoming NEET (Not in Education, Employment, or Training)
    for local authorities in England for young people between 16 and 18 years old.
    The significance of this work lies not only in preventing educational
    disengagement and unemployment but also in safeguarding the mental well-being
    of these vulnerable individuals. Through the timely implementation of tailored
    interventions, we seek to empower the lives of the young people by ensuring
    they stay engaged in education or find gainful employment opportunities.

    **The NEETalert tool**

    This tool aims to predict individuals at the risk of becoming NEET
    (Not in Education, Employment, or Training) between 16 and 18 years.
    It was developed during the University of Warwick’s Data Science for
    Social Good - UK 2023 (DSSGx UK) in cooperation with *councils in Bradford,
    Buckinghamshire, Solihull and Wolverhampton*, and *the EY Foundation*.

    The tool uses information from different datasets centrally reported to
    the Department for Education (DfE) by schools, including the School Census,
    Attendance, Exclusions, Attainment and the National Client Caseload Information
    System (NCCIS), as well as centrally provided school performance and regional
    deprivation datasets . Please read the information within the instructions menu
    item to learn more about the data requirements for using the tool.

    Please note – the tool is a prototype and has been trained using data from
    Bradford City Council. As such, it currently uses nuances discovered in this
    data to deliver risk of NEET predictions. These nuances may not be appropriate
    to assess the risk of NEET outside of this council without further research and
    development.
    """

    st.subheader(" ", anchor=False)

    if st.session_state.use_synthetic_data == True:
        st.markdown(
            '''
            **Use synthetic data**

            You are currently using synthetic data to test the dashboard.
            Every individual's data, as seen in the results, has been synthetically
            created and they are NOT real people. School names are real, but the results
            shown DO NOT reflect reality. The synthetic data shown should only be used
            to understand the type of analysis and visualisation the NEETalert tool offers
            and NOT for actual predictions.
            '''
        )
        st.button(
            "Deactivate synthetic data",
            on_click=ut.set_data,
            args=["model"],
            type="primary",
            help="You can use synthetic data to try the dashboard",
        )
    else:
        st.markdown(
            "We have genereted synthetic data. You can use it to try the dashboard"
        )
        st.button(
            "Activate synthetic data",
            on_click=ut.set_data,
            args=["synthetic"],
            type="primary",
            help="You can use synthetic data to try the dashboard",
        )


# Run the Streamlit app
if __name__ == "__main__":
    main()
