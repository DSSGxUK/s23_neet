import io
import zipfile
from typing import Literal
import pandera as pa
import streamlit as st
import utils as ut
from utils import (
    DATASET_TYPES,
    literal_dataset_types,
    COHORTS,
    literal_cohorts,
)
from neet.data_sources.preprocessing_functions import read_csv_file
from neet.data_sources.schema import get_schema

st.set_page_config(
    page_title="Data & Predictions - NEETalert",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded",
)


def render_upload_form() -> None:
    """
    Renders the upload form and encapsulates data by using nested functions.
    The input widget for the form content use a shadow state, not the st.session_state,
    because sessions state is not persistant for input widgets and cannot be written back
    to file uploader widgets.
    """

    def reset_form() -> None:
        """
        Used for the Form reset button. Currently not really working.
        """
        st.session_state.uploader_current_step = 0
        del st.session_state.uploader_shadow_state
        del st.session_state.uploader_response

        # Also remove from st.session state. Complicated because of uploader_ prefix
        for key in st.session_state:
            item = key.replace("uploader_", "")
            if item in DATASET_TYPES.keys():
                del st.session_state[key]

    def set_form_step(action: Literal["Next", "Back", "Jump"], step=None) -> None:
        """
        Used to set the different steps of the upload form.

        Args:
            action: How to move to another step of the form.
            step: If action is Jump the step to jump to needs to be defined.
        """
        if action == "Next":
            st.session_state.uploader_current_step = (
                st.session_state.uploader_current_step + 1
            )
        if action == "Back":
            st.session_state.uploader_current_step = (
                st.session_state.uploader_current_step - 1
            )
        if action == "Jump":
            st.session_state.uploader_current_step = step

    def run_shadow_state() -> None:
        """
        Widget state in Streamlit is not saved persistent in session
        state. This function creates a shadow state for the form
        input elements.

        Expects that only valids values are saved in shadow state.
        """

        # Set shadow state.
        for key, val in st.session_state.items():
            if ("uploader_" in key) and (
                key
                not in [
                    "uploader_shadow_state",
                    "uploader_response",
                    "uploader_current_step",
                    "uploader_cohort",
                ]
            ):
                # Only valid values are written to shadow state, so no check is needed
                st.session_state.uploader_shadow_state[key] = val

        # Overwrite based on shadow state
        for key in st.session_state.uploader_shadow_state:
            if key not in ["uploader_cohort"]:
                st.session_state[key] = st.session_state.uploader_shadow_state[key]

    def set_response(
        response_type: Literal["error", "success"],
        message: str = "An unexpected error occured.",
    ) -> None:
        """
        Sets a "form_response" session state based on the passed values. To be used with validation.
        The response type is either "success" or "error".

        Args:
            response_type: Either "success" or "error".
            message: A message that is shown for the selected response type.
        Returns:
            None
        """
        response = {
            "type": response_type,
            "message": message,
        }

        # State is already initalized at this point
        st.session_state.uploader_response = response

    def check_form_ready() -> bool:
        """
        Checks if files for all datasets are in shadow state and if
        cohort is set.

        Returns:
            True if data is available and valid; False if not
        """
        keys = set([f"uploader_{i}" for i in DATASET_TYPES.keys()])

        # Remove not required NCCIS files from keys
        drop = {
            "uploader_nccis_mar_y12",
            "uploader_nccis_sep_y13"
        }
        keys.difference_update(drop)

        # Add the key for the cohort select
        keys.add("uploader_cohort")

        uploaded = set([i for i in st.session_state.uploader_shadow_state])

        return True if keys.issubset(uploaded) else False

    def process_form() -> None:
        """
        Valdiates the form and if validation is successfull the uploaded files get processed.
        """

        # Reset the form error state
        if st.session_state.uploader_response is not None:
            del st.session_state.uploader_response

        # Be sure that all files are available.
        if check_form_ready() is False:
            set_response("error", "Please upload all files and select a cohort.")
            return

        # Get form content from shadow state.
        shadow_state = st.session_state.uploader_shadow_state
        cohort = shadow_state["uploader_cohort"]

        # Add files to data_raw session state
        for key, value in shadow_state.items():
            item = key.replace("uploader_", "")

            # Only do checks for datasets not the cohort select
            if item in DATASET_TYPES.keys():
                year = (item.split("_", 1)[1]).lstrip("y")
                schema = get_schema(item.split("_", 1)[0])

                # Read file from streamlit UploadedFile. Buffer needs to
                # be reset. At this point the file is already validated.
                value.seek(0)
                # For nccis data we set the year to zero, becasue it  includes multiple years.
                if ("nccis" in item) or ("september-guarantee" in item):
                    df = read_csv_file(value, cohort=cohort, year=0, schema=schema)
                else:
                    df = read_csv_file(
                        value, cohort=cohort, year=int(year), schema=schema
                    )

                # Add file to state
                ut.add_file_to_data_raw(df, cohort, item)

        # Print a success message at the end
        set_response("success", "The upload for cohort " + cohort + " was successfull.")
        reset_form()

    def upload_and_validate_file(dataset_type: literal_dataset_types) -> None:
        """
        Combines a file uploader with validation feedback and if a file
        is already uploaded it just prints the filename. This prevents errors
        because session state cannot be written back to file_uploader widgets.

        Args:
            dataset_type which is a dict key to also get a label.
        """
        key = str("uploader_" + dataset_type)

        # Check is file is already uploaded
        if key in st.session_state.uploader_shadow_state:
            file = st.session_state.uploader_shadow_state[key]
            if file:
                st.write(DATASET_TYPES[dataset_type] + " *(" + file.name + ")*")
                return

        # Render file uploader
        file = st.file_uploader(
            DATASET_TYPES[dataset_type],
            type=["csv"],
        )

        # Read file and validate the content.
        if file is not None:
            # Get only the dataset name that the schema function expects
            schema = get_schema(dataset_type.split("_", 1)[0])

            # Catch schema errors if they happen and print an error message
            try:
                # Passing dummy data to the function so it works
                df = read_csv_file(file, cohort="dummy", year=0, schema=schema)
            except pa.errors.SchemaErrors as err:
                st.error(
                    "The file does not satisfy the requirements. Please upload a valid file. The table below has additional information about the required file format and failure cases.",
                    icon="🚨",
                )
                st.dataframe(
                    err.failure_cases, use_container_width=True, hide_index=True
                )
                return

            # Write file to shadow store on success
            st.session_state.uploader_shadow_state[key] = file

            # Re-run so the interface looks correct.
            st.experimental_rerun()

    def set_cohort_select() -> None:
        """ "Write the session state for the cohort select to the shadow state"""

        # Set session state, if value is not "Select"
        cohort = st.session_state.uploader_cohort

        if cohort != "Select":
            st.session_state.uploader_shadow_state["uploader_cohort"] = cohort

    def form_header():
        """Renders the header for our upload form."""

        if "uploader_cohort" in st.session_state.uploader_shadow_state:
            st.header(
                "Upload datsets for "
                + st.session_state.uploader_shadow_state["uploader_cohort"]
                + " cohort",
                anchor=False,
            )
        else:
            st.header("Upload datasets per cohort", anchor=False)
        st.markdown("\n")

        # determines button color which should be red when user is on that given step
        chrt_type = (
            "primary" if st.session_state.uploader_current_step == 0 else "secondary"
        )
        atta_type = (
            "primary" if st.session_state.uploader_current_step == 1 else "secondary"
        )
        attd_type = (
            "primary" if st.session_state.uploader_current_step == 2 else "secondary"
        )
        cen_type = (
            "primary" if st.session_state.uploader_current_step == 3 else "secondary"
        )
        exlu_type = (
            "primary" if st.session_state.uploader_current_step == 4 else "secondary"
        )
        nccis_type = (
            "primary" if st.session_state.uploader_current_step == 5 else "secondary"
        )

        step_cols = st.columns(6)
        # The "1." makes Attainment and Attendance render nicely
        step_cols[0].button(
            "Cohort",
            on_click=set_form_step,
            args=["Jump", 0],
            use_container_width=True,
            type=chrt_type,
        )
        step_cols[1].button(
            "1. Attainment",
            on_click=set_form_step,
            args=["Jump", 1],
            use_container_width=True,
            type=atta_type,
        )
        step_cols[2].button(
            "1. Attendance",
            on_click=set_form_step,
            args=["Jump", 2],
            use_container_width=True,
            type=attd_type,
        )
        step_cols[3].button(
            "Census",
            on_click=set_form_step,
            args=["Jump", 3],
            use_container_width=True,
            type=cen_type,
        )
        step_cols[4].button(
            "Exclusions",
            on_click=set_form_step,
            args=["Jump", 4],
            use_container_width=True,
            type=exlu_type,
        )
        step_cols[5].button(
            "NCCIS",
            on_click=set_form_step,
            args=["Jump", 5],
            use_container_width=True,
            type=nccis_type,
        )

    def form_body():
        """Renders the body of the upload form"""

        # Step 1: Cohort
        if st.session_state.uploader_current_step == 0:
            st.markdown("\n")
            st.markdown("\n")

            # Write back value from shadow state to the selectbox
            if "uploader_cohort" in st.session_state.uploader_shadow_state:
                shadow = st.session_state.uploader_shadow_state["uploader_cohort"]
                if shadow:
                    st.session_state.uploader_cohort = shadow

            st.selectbox(
                "For which cohort do you want to upload files?",
                options=["Select"] + COHORTS,
                index=0,
                key="uploader_cohort",
                on_change=set_cohort_select,
            )

        # Step 2: Attainment
        if st.session_state.uploader_current_step == 1:
            st.markdown("\n")
            st.markdown("\n")

            upload_and_validate_file("ks4_y11")

        # Step 3: Attendance
        if st.session_state.uploader_current_step == 2:
            st.markdown("\n")
            st.markdown("\n")

            upload_and_validate_file("attendance_y11")
            upload_and_validate_file("attendance_y10")
            upload_and_validate_file("attendance_y9")
            upload_and_validate_file("attendance_y8")
            upload_and_validate_file("attendance_y7")

        # Step 4: Census
        if st.session_state.uploader_current_step == 3:
            st.markdown("\n")
            st.markdown("\n")

            upload_and_validate_file("census_y11")
            upload_and_validate_file("census_y10")
            upload_and_validate_file("census_y9")
            upload_and_validate_file("census_y8")
            upload_and_validate_file("census_y7")

        # Step 5 Exclusions
        if st.session_state.uploader_current_step == 4:
            st.markdown("\n")
            st.markdown("\n")

            upload_and_validate_file("exclusions_y11")

        # Step 6: NCCIS
        if st.session_state.uploader_current_step == 5:
            st.markdown("\n")
            st.markdown("\n")

            """
            Upload all the NCCIS data that is available for this cohort.
            The tool does predictions for the next 6 month. That is why it
            is important to upload the most recent NCCIS datasets.
            """

            upload_and_validate_file("september-guarantee_y12")
            upload_and_validate_file("nccis_mar_y12")
            upload_and_validate_file("nccis_sep_y13")

        st.markdown("---")

        form_footer_container = st.empty()

        with form_footer_container.container():
            disable_back_button = (
                True if st.session_state.uploader_current_step == 0 else False
            )
            hide_next_button = (
                True if st.session_state.uploader_current_step == 5 else False
            )

            form_footer_cols = st.columns([1.5, 1, 1.5, 1, 1])

            # This is currently broken ...
            form_footer_cols[0].button("Reset Uploader", on_click=reset_form)
            form_footer_cols[3].button(
                "Back",
                on_click=set_form_step,
                args=["Back"],
                disabled=disable_back_button,
            )

            if hide_next_button == False:
                form_footer_cols[4].button(
                    "Next",
                    on_click=set_form_step,
                    args=["Next"],
                )
            elif hide_next_button == True:
                # Here we have to check if all files are in session state
                form_ready = check_form_ready()

                load_file = form_footer_cols[4].button(
                    "Upload",
                    type="primary",
                    on_click=process_form,
                    disabled=(not form_ready),
                    help="Please add all files before you can upload the data for this cohort.",
                )

        if (
            "uploader_response" in st.session_state
            and st.session_state.uploader_response is not None
        ):
            response = st.session_state.uploader_response

            with st.container():
                if response["type"] == "error":
                    st.error(response["message"], icon="🚨")
                elif response["type"] == "success":
                    st.success(response["message"], icon="✅")

    # Set relevant session states and shadow state.
    if "uploader_current_step" not in st.session_state:
        st.session_state.uploader_current_step = 0
    if "uploader_shadow_state" not in st.session_state:
        st.session_state.uploader_shadow_state = {}
    if "uploader_response" not in st.session_state:
        st.session_state.uploader_response = None

    run_shadow_state()

    # Hacky way to get a border for the form. Might Cause UI problems
    with st.expander("Upload student data", expanded=True):
        # Print shadow state for debugging.
        # st.write(st.session_state.uploader_shadow_state)

        form_header()
        form_body()


def render_school_performance_uploader() -> None:
    """Renders a form to upload school perfosrmance data"""

    def remove_schools_performance() -> None:
        """Remove the school performance data from the sessions state."""
        st.session_state.file_schools_performance = None


    with st.expander("Upload school performance data", expanded=True):
        st.header("School performance data")
        st.markdown(
            """Please upload information about each school to enhance the predictions.
            You can download the correct dataset from the Department of Eduction website."""
        )

        # Bail early if we already have a file
        if st.session_state.file_schools_performance is not None:
            file = (st.session_state.file_schools_performance)["file"]
            st.write(f"Uploaded file: *{file.name}*")
            st.button("Remove file", on_click=remove_schools_performance)
            return

        file = st.file_uploader(
            "Upload school performance data",
            type=["csv"],
            label_visibility="collapsed",
        )

        if file is not None:
            # Get only the dataset name that the schema function expects
            schema = get_schema("school-performance")

            # Catch schema errors if they happen and print an error message
            try:
                # Passing dummy data to the function so it works
                df = read_csv_file(file, cohort="dummy", year=0, schema=schema)
            except pa.errors.SchemaErrors as err:
                st.error(
                    "This file does not satisfy the requirements. Please upload a valid file. The table below has additional information about the required file format and failure cases.",
                    icon="🚨",
                )
                st.dataframe(
                    err.failure_cases, use_container_width=True, hide_index=True
                )
                return

            # Write file to state for UI and df for further processing
            st.session_state.file_schools_performance = {"file":file, "df":df}

            # Re-run so the interface looks correct.
            st.experimental_rerun()


def render_list_of_files() -> None:
    """
    Renders a list of the complete cohorts that were uploaded to the dashboard.
    """

    @st.cache_resource
    def download_raw_files(cohort: literal_cohorts) -> zipfile.ZipFile:
        """
        Creates a zip archive with the raw files for the cohort.
        Data is taken from session_state.

        Args:
            cohort: Cohort to download data for.

        Returns:
            The zipfile.ZipFile object.
        """
        data_raw = st.session_state.data_raw

        values = [v for k, v in data_raw.items() if v["cohort"] == cohort]

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            for v in values:
                df = v["data"]
                name = str(v["cohort"]) + "_" + str(v["dataset_type"])
                zipf.writestr(
                    name + ".csv", df.to_csv(index=False, compression="infer")
                )

        return zip_buffer

    def remove_cohort_files(cohort: literal_cohorts) -> None:
        """
        Removes all files for the cohort from data_raw session_state
        and from disk.
        """
        data_raw = st.session_state.data_raw

        values = [v for k, v in data_raw.items() if v["cohort"] == cohort]

        for value in values:
            dataset_type = value["dataset_type"]
            ut.remove_file_from_data_raw(cohort, dataset_type)

    st.header("Uploaded files")
    """
    Below you can find the cohorts available in the dashboard. Cohorts are
    based on Year 11.
    The correct prediction is automatticaly determined by the latest available NCCIS
    file. Predictions are only generated for cohorts with complete NCCIS data.
    """

    if not st.session_state.data_raw:
        st.warning("Please upload files.")
        return

    # Find cohorts in data raw
    files = st.session_state.data_raw

    cohorts = sorted({d.get("cohort") for d in files.values()}, reverse=True)

    for cohort in cohorts:
        # Check NCCIS data for cohort:
        nccis = {
            f["dataset_type"]
            for f in files.values()
            if f["cohort"] == cohort
            and ("nccis" in f["dataset_type"])
            or ("september-guarantee" in f["dataset_type"])
        }

        if len(nccis) == 4:
            output = "Complete cohort"
        elif 1 <= len(nccis) <= 3:
            output = ", ".join([DATASET_TYPES[f] for f in nccis])
        else:
            output = "Error. Please remove this cohort!"

        name, remove, download = st.columns([2, 1, 1])
        name.markdown(f"**Cohort {cohort}** <br> *{output}*", unsafe_allow_html=True)
        remove.button(
            "Delete Cohort",
            key=cohort + "_delete",
            on_click=remove_cohort_files,
            args=[cohort],
            help="Be careful! This action is irreversible!",
        )

        zip_file = download_raw_files(cohort).getvalue()

        download.download_button(
            "Download",
            data=zip_file,
            file_name=cohort + ".zip",
            mime="application/zip",
            key=cohort + "_download",
        )


@st.cache_data
def create_final_csv():
    """
    Create the final csv file with all predictions.
    """
    data = st.session_state.data_final
    return data.to_csv(index=False, compression="infer")


def main():
    """Renders the contents of the streamlit page."""

    # Add global styles and load state

    ut.initalize_global_state()
    ut.initalize_data_raw_state()

    st.title("Data and Predictions", anchor=False)

    render_upload_form()

    render_school_performance_uploader()

    render_list_of_files()

    st.header("Process data & calculate predictions", anchor=False)
    """
    A prepared machine learning model is used to make predictions about NEET individuals
    based on the provided data. This model is fast and should creat valid predictions.
    Nonetheless extreme care as to be taken in evaluating the results. Please consult the
    provided documentation and instructions for futher information.
    """
    col1, col2 = st.columns(2)

    col1.button(
        "Process data & calculate predictions",
        type="primary",
        on_click=ut.calculate_predictions,
        # Only run if raw data is available
        disabled=(
            len(st.session_state.data_raw) == 0
            or not st.session_state.file_schools_performance
        ),
    )

    # Only show button if data_final exists
    if "data_final" in st.session_state:
        final_csv = create_final_csv()

        col2.download_button(
            label="Download predictions",
            data=final_csv,
            file_name="NEET_Risk_Predictions.csv",
            mime="text/csv",
        )

        # st.info(
        #     f"Predictions done for X out of Y students. Z were dropped because of missing data."
        # )

# Run the Streamlit app
if __name__ == "__main__":
    main()
