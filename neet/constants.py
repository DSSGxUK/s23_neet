import os
from pathlib import Path
from dotenv import load_dotenv
from typing import NamedTuple, List
import pandas as pd

load_dotenv()

########
# DATA #
########
DATA_DEVELOP_PATH = Path.cwd() / os.getenv("DATA_DEVELOP_PATH")
DATA_STREAMLIT_PATH = Path.cwd() / os.getenv("DATA_STREAMLIT_PATH")

# Data develop
DATA_DEVELOP_RAW_PATH = DATA_DEVELOP_PATH / "raw"
DATA_DEVELOP_INTERMEDIATE_PATH = DATA_DEVELOP_PATH / "intermediate"
DATA_DEVELOP_FINAL_PATH = DATA_DEVELOP_PATH / "final"

# create folders only if they not exists 
# (temporary solution to avoid FileNotExist error)
DATA_DEVELOP_RAW_PATH.mkdir(parents=True, exist_ok=True)
DATA_DEVELOP_INTERMEDIATE_PATH.mkdir(parents=True, exist_ok=True)
DATA_DEVELOP_FINAL_PATH.mkdir(parents=True, exist_ok=True)

# Data Streamlit 
DATA_STREAMLIT_RAW_PATH = DATA_STREAMLIT_PATH / "raw"
DATA_STREAMLIT_INTERMEDIATE_PATH = DATA_STREAMLIT_PATH / "intermediate"
DATA_STREAMLIT_FINAL_PATH = DATA_STREAMLIT_PATH / "final"

# create folders only if they not exists 
# (temporary solution to avoid FileNotExist error)
DATA_STREAMLIT_RAW_PATH.mkdir(parents=True, exist_ok=True)
DATA_STREAMLIT_INTERMEDIATE_PATH.mkdir(parents=True, exist_ok=True)
DATA_STREAMLIT_FINAL_PATH.mkdir(parents=True, exist_ok=True)

STREAMLIT_PATH = Path.cwd() / "neet" / "streamlit_api"

#############
# RANDOM STATE #
#############
SEED = 1234  # Not used

########
# SYNTHETIC DATA #
########

DATA_SYNTHETIC = Path.cwd() / "neet" / "assets" / "data_synthetic" / "synthetic.csv"


###########
# CLASSES #
###########
class DatasetType(NamedTuple):
    attendance: List[pd.DataFrame]
    census: List[pd.DataFrame]
    exclusions: List[pd.DataFrame]
    ks4: List[pd.DataFrame]
    nccis: List[pd.DataFrame]
    september_guarantee: List[pd.DataFrame]
    # school_performance: List[pd.DataFrame]
    #regional_data :List[pd.DataFrame]
    #postcodes : List[pd.DataFrame]
