import streamlit as st
import utils as ut

st.set_page_config(
    page_title="Instructions - NEETalert",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded",
)


def main():
    """Renders the contents of the streamlit page."""

    # Add global styles and load state

    ut.initalize_global_state()

    st.title("Instructions")

    st.header("About the Data")

    st.markdown(
        '''
        The NEETalert tool is dependent on a variety of different datasets.
        The more **complete** or **comprehensive** the data collated within each dataset is,
        the **better the performance of the tool**, in terms of provision of risk of NEET predictions.

        Please follow the instructions to provide the correct data formatting for the tool to work.
        '''
        )


    st.header("Datasets")

    with st.expander("NCCIS Data"):
        st.markdown(
            '''
            National Client Caseload Information System (NCCIS) data is submitted to
            the Department for Education(DfE) by the local authorities.
            It monitors and records the extent to which an individual is involved in
            education and training. It is the file which contains the target variable
            for our prediction model (through the activity codes).

            The data uploader expects each NCCIS file to contain the following columns of
            data as a minimum:

            stud_id, age, academic_age, support_level, looked_after_in_care, caring_for_own_child,
            refugee_asylum_seeker, carer_not_own_child, substance_misuse, care_leaver, supervised_by_yots,
            pregnancy, parent, teenage_mother, send, alternative_provision, sensupport,
            confirmed_date, postcode, nccis_code, ncciscohort

            '''
        )
    with st.expander("School Census Data"):
        st.markdown(
            '''
            This data provides demographic information about individuals such as gender,
            ethnicity, age, language, eligibility for Free School Meals (FSMs) or
            Special Educational Needs (SENs).

            The data uploader expects each School Census file to contain the following
            columns of data as a minimum:

            stud_id, date_of_birth, forename, surname, estab, gender, entry_date,
            ncyear_actual, ethnicity, language, senprovision, senneed1, senneed2, senunit_indicator,
            resourced_provision_indicator, fsme_on_census_day, age
            '''
        )
    with st.expander("Attainment Data"):
        st.markdown(
            '''
            It holds information related to the individual’s grades and various attainment scores.

            The data uploader expects each Attainment file to contain the following columns
            of data as a minimum:

            stud_id, ks4_acadyr, ks4_yeargrp, ks4_actyrgrp, ks4_la, ks4_estab, ks4_att8,
            ks4_pass_94, ks4_priorband_ptq_ee'''
        )
    with st.expander("Attendance Data"):
        st.markdown(
            '''
            This data captures the attendance of individuals along with features as termly sessions,
            absences, and reasons for absences, e.g. exclusions, late entries etc.

            The data uploader expects each Attendance file to contain the following columns
            of data as a minimum:

            stud_id, possible_sessions, attendance_count, authorised_absence, unauthorised_absence,
            excluded_e_count
            '''
        )

    with st.expander("Exclusions Data"):
        st.markdown(
            '''
            This data captures the information about an individual’s historical exclusion status.

            The data uploader expects each Exclusions file to contain the following columns
            of data as a minimum:

            stud_id, ever_suspended, ever_excluded, exclusions_rescinded

            '''
        )


# Run the Streamlit app
if __name__ == "__main__":
    main()
