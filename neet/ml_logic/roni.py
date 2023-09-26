import numpy as np
import pandas as pd


def calculate_roni_score(
    df: pd.DataFrame, threshold: int = 2, nccis: bool = False
) -> pd.DataFrame:
    """
    Calculates a roni risk score for each student based on which risk factors they have.

    Roni tool risk factors, score given if the student has that characteristic
    and columns from which risk factor is measured in our dataset.
    This functions works with or without the availability of NCCIS data. Without NCCIS
    columns a lower thresholds needs to be selected.

    Threshold suggestions:
    4+ = Substantial support needed to avoid becoming NEET.
    2-3 = Additional support needed
    1 = Additional support should be offered (but may not beneeded

    Scoring is based on:
    https://schools.oxfordshire.gov.uk/cms/sites/schools/files/folders/folders/documents/careersadvice/Oxfordshire_RONI_criteria.pdf

    Args:
        df: modelling dataframe indexed by stud_id
        threshold: threshold as to which a student is classified as NEET.
                   e.g. if threshold = 3, every student with a roni score of 3 will be high-risk
        nccis: If the NCCIS data should be used or not.

    Return
        pd.DataFrame with columns for each roni tool risk factor weighting and the overall roni score for each student
    """
    # Initalize empty dataframe for results
    rdf = pd.DataFrame()

    # Attendance <90%, weighting = 1, Attendance <85%, weighting = 2 (coded as 1+1)
    rdf["attendance_85"] = np.where(
        (df["attendance_count_11"] / df["possible_sessions_11"]) <= 0.85, 1, 0
    )
    rdf["attendance_90"] = np.where(
        (df["attendance_count_11"] / df["possible_sessions_11"]) <= 0.9, 1, 0
    )

    # English as additional language, weighting = 1
    rdf["english"] = np.where(df["census_language"] != "ENG", 1, 0)

    # SEND means student has EHC plan, weighting = 2,
    rdf["send"] = np.where(df["census_senprovision_y11"] == "E", 2, 0)

    # Special Educational Needs (SEN), weighting = 1
    rdf["sensupport"] = np.where(df["census_senprovision_y11"] == "K", 1, 0)

    # Exclusions, but we cannot weight by time excluded, weighting = 2
    rdf["ever_excluded"] = np.where(df["excluded_ever_excluded"] == True, 2, 0)

    # Eligible for Free School Meals, weighting = 2
    rdf["fsme_y11"] = np.where(df["census_fsme_on_census_day"] > 0, 2, 0)

    # The following indicators are only available in the NCCIS data.
    if nccis:
        # Educated at Alternative Provision, weighting = 1
        rdf["alternative_provision"] = np.where(
            df["nccis_alternative_provision"] == True, 1, 0
        )

        # Looked-after, weighting = 2
        rdf["looked_after_in_care"] = np.where(df["nccis_looked_after_in_care"] == True, 2, 0)

        # Pregnant or parent (only if they care for child), weighting = 2
        rdf["pregnant_parent"] = np.where(
            df[["nccis_parent", "nccis_pregnancy"]].eq(True).any(axis=1), 2, 0
        )

        # Carer, weighting = 2
        rdf["carer_not_own_child"] = np.where(df["nccis_carer_not_own_child"] == True, 2, 0)

        # Custody, weighting = 2
        rdf["custody"] = np.where(df["nccis_code"] == 710, 2, 0)

        # Supervised by Youth Offending Team, weighting = 2
        rdf["supervised_by_yots"] = np.where(df["nccis_supervised_by_yots"] == True, 2, 0)

    # Add columns to the dataframe
    rdf["roni_score"] = rdf.sum(axis=1)
    rdf["roni_classification"] = np.where(rdf["roni_score"] > threshold, True, False)

    # Add the stud_id
    rdf["stud_id"] = df.index

    # Keep only certain score, classification and id for return
    return (
        rdf[["roni_score", "roni_classification", "stud_id"]]
        .set_index("stud_id")
        .sort_index()
    )
