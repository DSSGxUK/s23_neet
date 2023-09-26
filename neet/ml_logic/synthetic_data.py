from imblearn.over_sampling import SMOTE, SMOTENC
import pandas as pd
from sklearn.utils import shuffle
import data_imputations as di
import neet.data_sources.feature_extraction_functions as f
import neet.data_sources.constants as constants
import os
import neet.data_sources.datacleaning_functions as dcf

df= pd.read_parquet(os.getenv("INTERMEDIATE_PREPROC_MODEL2"),engine='pyarrow')
df = dcf.mapping_target_varible(df)
df = dcf.encode_and_data_type_conversions(df,constants.COLUMNS_CATEGORICAL_NOMINAL_MODEL2, 
                                             constants.COLUMNS_CATEGORICAL_ORDINAL_MODEL2,
                                             constants.COLUMNS_NUMERIC_MODEL2)

columns_to_drop= ['nccis_confirmed_date', 
'census_senneed1_y11','census_senneed2_y11','census_surname','census_forename',
'attendance_count_7','unauthorised_absence_7','possible_sessions_7','authorised_absence_7',
'home_latitude','home_longitude',
'school_latitude','school_longitude',
'nccis_academic_age',
'nccis_order_of_nccis_update','census_cohort',
'ks4_cohort', 'ks4_fsm','ks4_fsm6',
'nccis_confirmed_date',
'census_estab','ks4_estab', 'excluded_year',
'excluded_cohort','school_postcode', 
'lsoa_name_2011','postcode','nccis_time_recorded',
'nccis_parent'
]

df = df.drop(columns=columns_to_drop)

df = df.dropna() 
print(f'The shape of the dataset is {df.shape}')

#df['neet_label'] = df['nccis_status'].apply(neet_mapping)


#SMOTE to handle imbalance
cat_vars = df.select_dtypes(include=[ 'category']).columns
cat_vars = cat_vars.values.tolist()


for col in constants.COLUMNS_TO_IMPUTE_MEAN_ATTENDANCE:
    try:
        df[col]= df[col].astype(int)
    except KeyError:
        pass

#df.to_parquet("/home/workspace/files/intermediate/Testingfiles/test_file.parquet")


# Assuming you have a DataFrame named 'df' and a column named 'neet_label'
X = df.drop('nccis_status', axis=1)
y = df['nccis_status'].astype("category")


smote = SMOTENC(categorical_features = cat_vars,random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X,y)

df = pd.concat([X_resampled,y_resampled], axis =1)

# Create a new column to indicate whether a sample is synthetic or not
df['is_original'] = df.index.isin(X.index)

print(pd.unique(df['is_original']))

original = df[df['is_original'] == True]
combined_synthetic_data = df[df['is_original'] == False]

# Combine the synthetic data
#combined_synthetic_data = pd.concat([synthetic_X, synthetic_y], axis=1)

# Step 1: Select 95% of the non-NEET and 5% of the NEET samples from the combined synthetic data
non_neet_samples = combined_synthetic_data[combined_synthetic_data['nccis_status'] == 0]
neet_samples = combined_synthetic_data[combined_synthetic_data['nccis_status'] == 1]

# Calculate the number of samples to select based on the percentages
num_non_neet_samples = int(0.95 * len(non_neet_samples))
num_neet_samples = int(0.05 * len(neet_samples))

# Randomly sample the required number of non-NEET and NEET samples
selected_non_neet_samples = non_neet_samples.sample(n=num_non_neet_samples, random_state=42)
selected_neet_samples = neet_samples.sample(n=num_neet_samples, random_state=42)

# Step 2: Form a new DataFrame 'final_data' from the selected samples
final_data = pd.concat([selected_non_neet_samples, selected_neet_samples])

# Add the columns that were deleted
dropped_df = pd.DataFrame(columns=columns_to_drop)


final_data = pd.concat([final_data,dropped_df],axis=1)


# Display the shape of the final_data DataFrame
print("The number of Non NEET students are", selected_non_neet_samples.shape)
print("The number of NEET students are",  selected_neet_samples.shape)

print("Shape of final_data:", final_data.shape)

#Save final synthetic data
final_data.to_csv("/home/workspace/files/intermediate/synthetic_data/synthetic_data.csv", index = False)