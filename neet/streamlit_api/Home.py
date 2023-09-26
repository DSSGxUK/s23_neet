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

    st.title("Welcome to NEETalert! 👋", anchor=False)
    """
    This tool aims to predict young persons at the risk of becoming NEET (Not 
    in Education, Employment, or Training) between 16 and 18 years. It was developed 
    during Data Science for Social Good - UK 2023 (DSSGx UK) in cooperation with 
    Bradford Council, Buckinghamshire Council, Solihull Council, Wolverhampton Council and 
    the EY Foundation. 
    
    The tool uses information from different datasets, inclidung thethe School Census, 
    Key Stage 4 results and the National Client Caseload Information System (NCCIS). 
    Please read the documentation to learn more about our methodology and how to use the tool
    for your Council.
    """

    st.subheader("Use synthetic data", anchor=False)

    if st.session_state.use_synthetic_data == True:
        st.markdown(
            "You are currently using synthetic data to test the dashboard. You can deactivate it below."
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
