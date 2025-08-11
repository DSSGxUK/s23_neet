COLUMNS_ATTENDANCE = {"possible_sessions", "attendance_count", "authorised_absence", 
                     "unauthorised_absence", "stud_id", "cohort",'year'}

COLUMNS_NCCIS = {'academic_age', 'support_level',
                 'looked_after_in_care','caring_for_own_child', 'refugee_asylum_seeker', 
                 'carer_not_own_child','substance_misuse', 'care_leaver', 'supervised_by_yots',
                 'pregnancy','parent', 'teenage_mother', 'send', 'alternative_provision',
               'sensupport', 'confirmed_date', 'postcode',
                'nccis_code'}

COLUMNS_SEPTEMBER_GUARANTEE = {'academic_age', 'support_level',
                 'looked_after_in_care','caring_for_own_child', 'refugee_asylum_seeker', 
                 'carer_not_own_child','substance_misuse', 'care_leaver', 'supervised_by_yots',
                 'pregnancy','parent', 'teenage_mother', 'send', 'alternative_provision',
               'sensupport', 'confirmed_date', 'postcode'
               }

#COLUMNS_KS2 = {"examyear_re"
#    ,"examyear_gps"
#    ,"acadyr"
#    ,"yeargrp"
#    ,"la"
#    ,"examyear_ma"
#    ,"readmrk"
#    ,"cla_pp_1_day"
#    ,"cla_pp_6_months"
#    ,"ks1group"
#    ,"fsm"
#    ,"fsm6"
#    ,"fsm6_p"
#    ,"nftype"
#    ,"readscore"
#    ,"matscore"
#    ,"gpsscore"
#    ,"ks1readps"
#    ,"ks1writps"
#    ,"ks1matps"
#    ,"ks1average"
#    ,"sentype" }
#

COLUMNS_KS2 = {'stud_id'}

#COLUMNS_KS4 = {"ks4_acadyr", 
#        "stud_id",
#        "ks4_la",
#        "ks4_la_9code",
#        "ks4_estab",
#        "ks4_laestab",
#        "ks4_pass_94",
#        "ks4_pass_1_94",
#        "ks4_glevel2_ptq_ee",
#        "ks4_glevel2em_ptq_ee",
#        "ks4_anylev1_ptq_ee",
#        "ks4_anypass_ptq_ee",
#        "ks4_hgmath_91",
#        "ks4_lev2em_94",
#        "ks4_lev2em_95",
#        "ks4_lev1em_ptq_ee",
#        "ks4_ebaceng_94",
#        "ks4_ebaceng_95",
#        "ks4_priorband_ptq_ee",
#        "ks4_ealgrp_ptq_ee",
#        "ks4_apeng_91",
#        "ks4_fsm",
#        "ks4_fsm6",
#        "ks4_fsm6_p",
#        "ks4_senps",
#        "ks4_sena",
#        "ks4_flang",
#        "ks4_att8",
#        "ks4_gcse_1",
#        "ks4_gcse_2",
#        "ks4_gcse_3",
#        "ks4_gcse_4",
#        "ks4_gcse_5",
#        "ks4_gcse_6",
#        "ks4_gcse_7",
#        "ks4_gcse_8",
#        "ks4_gcse_9",
#        "ks4_gcse_91",
#        "ks4_gcse_94",
#        "ks4_gcse_95",
#        "ks4_five91",
#        "ks4_five94",
#        "ks4_pass_1_91",
#        "ks4_ks2readscoreta",
#        "ks4_ks2matscoreta",
#        "ks4_ks2readscore",
#        "ks4_ks2matscore",
#        "ks4_ks2emss",
#        "ks4_ks2emss_grp",
#        "ks4_idaci"}

COLUMNS_KS4 = {"stud_id","ks4_priorband_ptq_ee",
        "ks4_att8",'ks4_estab','ks4_pass_94'}

COLUMNS_REGIONAL = {'postcode','lsoa_name_2011',
    'index_of_multiple_deprivation_imd_score',
    'income_score_rate','employment_score_rate',
    'education_skills_and_training_score',
    'health_deprivation_and_disability_score',
    'crime_score','living_environment_score',
    'income_deprivation_affecting_children_index_idaci_score_rate'}

COLUMNS_SCHOOL_PERFORMANCE = {'ks4_estab',
        'school_postcode',
        'perc_overall_abscence',
        'perc_of_girls',
        'perc_of_eligible_sen_pupil',
        'perc_english_flang',
        'total_fsm_ever', 
        'ofstedrating',
        'schooltype_y'
        }

COLUMNS_CENSUS = {'stud_id', 'mob', 'estab', 'gender', 'enrol_status', 'entry_date',
       'ncyear_actual', 'ethnicity', 'language', 'senprovision', 'senneed1',
       'senneed2', 'senunit_indicator', 'resourced_provision_indicator',
       'fsme_on_census_day', 'age', 'cohort', 'year'}

COLUMNS_POSTCODES = {'postcode', 'latitude' ,'longitude'}

EXCEL_FILES_ATTENDANCE = {"RONI UPNs 2018_19.xlsx":{"sheets":["Y11 201819 Attend", "Y11 201718 Attend", "Y11 201617 Attend", "Y11 201516 Attend"], 
                                         "cohort":["2018-19", "2018-19", "2018-19", "2018-19"],
                                         "years": [11, 10, 9, 8]},
               "RONI UPNs 2019_20.xlsx":{"sheets":["Y11 201920 Attend", "Y11 201819 Attend", "Y11 201718 Attend", "Y11 201617 Attend", "Y11 201516 Attend"], 
                                         "cohort":["2019-20", "2019-20", "2019-20", "2019-20", "2019-20"],
                                         "years": [11, 10, 9, 8, 7]},
               "RONI UPNs 2020_21.xlsx":{"sheets":["Y11 202021 Attend", "Y11 201920 Attend", "Y11 201819 Attend", "Y11 201718 Attend", "Y11 201617 Attend"], 
                                         "cohort":["2020-21", "2020-21", "2020-21", "2020-21", "2020-21"],
                                         "years": [11, 10, 9, 8, 7]},
               "RONI UPNs 2021_22.xlsx":{"sheets":["Y11 202122 Attendance 21_22", "Y11 202122 Attendance 20_21", "Y11 202122 Attendance 19_20", "Y11 202122 Attendance 18_19", "Y11 202122 Attendance 17_18"], 
                                         "cohort":["2021-22", "2021-22","2021-22", "2021-22", "2021-22"],
                                         "years": [11, 10, 9, 8, 7]}
              }

EXCEL_FILES_CENSUS = {"Y11 UPNs 2018_19 Census Data.xlsx":{"sheets":["Y11 201819 Jan 19 Census", "Y10 201819 Jan 18 Census", "Y9 201819 Jan 17 Census", "Y8 201819 Jan 16 Census"], 
                                         "cohort":["2018-19", "2018-19", "2018-19", "2018-19"],
                                         "years": [11, 10, 9, 8]},

               "Y11 UPNs 2019_20 Census Data.xlsx":{"sheets":["Y11 201920 Jan Census 20", "Y10 201920 Jan Census 19", "Y9 201920 Jan Census 18", "Y8 201920 Jan Census 17", "Y7 201920 Jan Census 16"], 
                                         "cohort":["2019-20", "2019-20", "2019-20", "2019-20", "2019-20"],
                                         "years": [11, 10, 9, 8, 7]},

               "Y11 UPNs 2020_21 Census Data.xlsx":{"sheets":["Y11 202021 Jan 21 Census", "Y10 202021 Jan 20 Census", "Y9 202021 Jan 19 Census", "Y8 202021 Jan 18 Census", "Y7 202021 Jan 17 Census"], 
                                         "cohort":["2020-21", "2020-21", "2020-21", "2020-21", "2020-21"],
                                         "years": [11, 10, 9, 8, 7]},

               "Y11 UPNs 2021_22 Census Data.xlsx":{"sheets":[' Y11 202122 Jan 22 Cens', ' Y10 202021 Jan 21 Cens', ' Y9 201920 Jan 20 Cens', ' Y8 201819 Jan 19 Cens', ' Y7 201718 Jan 18 Cens'] , 
                                         "cohort":["2021-22", "2021-22","2021-22", "2021-22", "2021-22"],
                                         "years": [11, 10, 9, 8, 7]}
              }


#EXCEL_FILES_CENSUS = {"RONI UPNs 2018_19.xlsx":{"sheets":["Y11 201819 Census"], "cohort":["2018-19"], "years":[11]},
#               "RONI UPNs 2019_20.xlsx":{"sheets":["Y11 201920 Census"], "cohort":["2019-20"], "years":[11]},
#               "RONI UPNs 2020_21.xlsx":{"sheets":["Y11 202021 Census"], "cohort":["2020-21"], "years":[11]},
#               "RONI UPNs 2021_22.xlsx":{"sheets":[" Y11 202122 Jan Cens"], "cohort":["2021-22"], "years":[11]}}

EXCEL_FILE_EXCLUSIONS = {"RONI UPNs 2018_19.xlsx":{"sheets":["Y11 201819 Exclude"], "cohort":["2018-19"], "years":[11]},
               "RONI UPNs 2019_20.xlsx":{"sheets":["Y11 201920 Exclude"], "cohort":["2019-20"], "years":[11]},
               "RONI UPNs 2020_21.xlsx":{"sheets":["Y11 202021 Exclude"], "cohort":["2020-21"], "years":[11]},
               "RONI UPNs 2021_22.xlsx":{"sheets":[" Y11 202122 Exclude"], "cohort":["2021-22"], "years":[11]}}

EXCEL_FILES_NCCIS = {"Act16to19IMT Mar 19.xlsx":{"sheets":[0], "cohort":["Mar-19"], "years":[0]}, 
               "Act16to19IMT Mar 20.xlsx":{"sheets":[0], "cohort":["Mar-20"], "years":[0]}, 
               "Act16to19IMT Mar 21.xlsx":{"sheets":[0], "cohort":["Mar-21"], "years":[0]}, 
               "Act16to19IMT Mar 22.xlsx":{"sheets":[0], "cohort":["Mar-22"], "years":[0]},
               "Act16to19IMT Mar 23.xlsx":{"sheets":[0], "cohort":["Mar-23"], "years":[0]}}
              
CSV_FILES_SEPTEMBER_GUARANTEE = {'Act16to19 Sep 22 UL.csv':{"sheets":[0], "cohort":["Sep-22"], "years":[0]}, 
               'Act16to19IMT Sep 18 UL.csv':{"sheets":[0], "cohort":["Sep-18"], "years":[0]},
               'Act16to19IMT Oct 19 UL.csv':{"sheets":[0], "cohort":["Sep-19"], "years":[0]},
               'Act16to19IMT Sep 20 UL.csv':{"sheets":[0], "cohort":["Sep-20"], "years":[0]}, 
               'Act16to19IMT Sep 21 UL.csv':{"sheets":[0], "cohort":["Sep-21"], "years":[0]}}

EXCEL_FILES_KS2 = {"Keystage Results 2018_19.xlsx":{"sheets":["KS2"], "cohort":["2018-19"], "years":[6]},
               "Keystage Results 2019_20.xlsx":{"sheets":["KS2"], "cohort":["2019-20"], "years":[6]}, 
               "Keystage Results 2020_21.xlsx":{"sheets":["KS2"], "cohort":["2020-21"], "years":[6]}, 
               "Keystage Results 2021_22.xlsx":{"sheets":["KS2"], "cohort":["2021-22"], "years":[6]}}

EXCEL_FILES_KS4 = {"Keystage Results 2018_19.xlsx":{"sheets":["KS4"], "cohort":["2018-19"], "years":[11]},
               "Keystage Results 2019_20.xlsx":{"sheets":["KS4"], "cohort":["2019-20"], "years":[11]}, 
               "Keystage Results 2020_21.xlsx":{"sheets":["KS4"], "cohort":["2020-21"], "years":[11]}, 
               "Keystage Results 2021_22.xlsx":{"sheets":["KS4"], "cohort":["2021-22"], "years":[11]}}


EXCEL_FILES_REGIONAL_DATA = {"IDS_2019.xlsx":{"sheets":["IoD2019 Scores"],"cohort":["2019"], "years":[11]}}
CSV_FILES_REGIONAL_DATA = {"pcd_to_lsoa_codes.csv":{"sheets":["PCD to LSOA"],"cohort":["2019"], "years":[11]}}

EXCEL_FILES_SCHOOL_PERFORMANCE = {"additional_data_part2.xlsx":{"sheets":[0], "cohort":["2021-22"], "years":[11]}}

CSV_FILES_POSTCODES = {"ukpostcodes.csv":{"sheets":["ukpostcodes"],"cohort":["2000"], "years":[11]}}

EXCLUDE_COLUMN_RENAME = ['stud_id','postcode','ks4_estab','nccis_code']


# Columns for data type conversions For Model 1
COLUMNS_CATEGORICAL_NOMINAL_MODEL1 = ['september_guarantee_carer_not_own_child','september_guarantee_send',
                                      'september_guarantee_teenage_mother','september_guarantee_alternative_provision',
                                'postcode','september_guarantee_parent','september_guarantee_sensupport',
                                'september_guarantee_pregnancy','september_guarantee_care_leaver','september_guarantee_substance_misuse',
                                'september_guarantee_looked_after_in_care','september_guarantee_refugee_asylum_seeker','september_guarantee_caring_for_own_child',
                                'september_guarantee_supervised_by_yots','september_guarantee_academic_age','census_estab','census_gender',
                                'census_ethnicity','census_language','census_senprovision_y11',
                                'ks4_estab',
                                'excluded_ever_suspended','excluded_ever_excluded','excluded_exclusions_rescinded','ofstedrating','schooltype_y']

COLUMNS_CATEGORICAL_ORDINAL_MODEL1 = ['september_guarantee_support_level']


COLUMNS_NUMERIC_MODEL1 = ['stud_id','census_fsme_on_census_day','census_resourced_provision_indicator','census_senunit_indicator','ks4_pass_94']


#Columns for data type conversions For Model 2
COLUMNS_CATEGORICAL_NOMINAL_MODEL2 = ['nccis_carer_not_own_child','nccis_send',
                                      'nccis_teenage_mother','nccis_alternative_provision',
                                'postcode','nccis_parent','nccis_sensupport','nccis_pregnancy','nccis_care_leaver','nccis_substance_misuse',
                                'nccis_looked_after_in_care','nccis_refugee_asylum_seeker','nccis_caring_for_own_child',
                                'nccis_supervised_by_yots','nccis_academic_age','census_estab','census_gender',
                                'census_ethnicity','census_language',
                                 'ks4_estab', 'census_senprovision_y11',
                                'excluded_ever_suspended','excluded_ever_excluded','excluded_exclusions_rescinded','ofstedrating' , 
                                'schooltype_y']

COLUMNS_CATEGORICAL_ORDINAL_MODEL2 = ['nccis_support_level']


COLUMNS_NUMERIC_MODEL2 = ['stud_id','census_fsme_on_census_day','census_resourced_provision_indicator','census_senunit_indicator','ks4_pass_94']

# Mean value imputations:

COLUMNS_TO_IMPUTE_MEAN_ATTENDANCE =['attendance_count_11','unauthorised_absence_11','possible_sessions_11','authorised_absence_11','attendance_count_8',
      'unauthorised_absence_8','possible_sessions_8','authorised_absence_8','attendance_count_9','unauthorised_absence_9',
     'possible_sessions_9','authorised_absence_9','attendance_count_10','unauthorised_absence_10','possible_sessions_10',
     'authorised_absence_10','attendance_count_7','unauthorised_absence_7','possible_sessions_7','authorised_absence_7']

# Can be removed if we find a better way to impute below values:
#EXTRA_COLUMNS_FOR_MEAN_IMPUTATIONS = ['health_deprivation_and_disability_score','income_score_rate',
#                                     'index_of_multiple_deprivation_imd_score','crime_score',
#                                     'income_deprivation_affecting_children_index_idaci_score_rate',
#                                     'living_environment_score','education_skills_and_training_score',
#                                     'employment_score_rate']

COLUMNS_N_Y_TYPE = ['nccis_carer_not_own_child','nccis_send','nccis_teenage_mother','nccis_alternative_provision','nccis_sensupport',
 'nccis_pregnancy','nccis_care_leaver','nccis_substance_misuse','nccis_looked_after_in_care','nccis_refugee_asylum_seeker',
 'nccis_caring_for_own_child','nccis_supervised_by_yots','census_senunit_indicator','census_resourced_provision_indicator',
 'census_fsme_on_census_day','excluded_ever_suspended','excluded_ever_excluded', 'census_senprovision',
 'excluded_exclusions_rescinded', 'september_guarantee_carer_not_own_child','september_guarantee_send',
 'september_guarantee_teenage_mother','september_guarantee_alternative_provision',
 'september_guarantee_sensupport',
 'september_guarantee_pregnancy','september_guarantee_care_leaver',
 'september_guarantee_substance_misuse','september_guarantee_looked_after_in_care',
 'september_guarantee_refugee_asylum_seeker',
 'september_guarantee_caring_for_own_child',
 'september_guarantee_supervised_by_yots']

