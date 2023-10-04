# Predicting students at risk of becoming NEET (Not in Education, Employment or Training)

Welcome to the code repository for the project conducted under Data Science for Social Good - UK 2023 (DSSGx UK).
This README provides an overview of the project, the contributors, the folder structure of the repository, and guidance on how other councils can replicate the project using their own datasets.


This project aims to develop a predictive model to identify individuals at risk of becoming NEET (Not
in Education, Employment, or Training) for local authorities in England for young people between 16
and 18 years old. The significance of this work lies not only in preventing educational disengagement and
unemployment but also in safeguarding the mental well-being of these vulnerable individuals. Through
the timely implementation of tailored interventions, we seek to empower the lives of the young people
by ensuring they stay engaged in education or find gainful employment opportunities.

<br>

<details>
<summary>  <h1> Introduction </h1> </summary>

The transition from adolescence to adulthood is a critical phase that shapes an individual’s future prospects,
impacting their education, employment, and overall well-being. Among the challenges that young people
face during this transition is the risk of being categorized as ”NEET” – Not in Education, Employment, or
Training. The NEET status has garnered significant attention due to its association with adverse outcomes,
particularly in terms of mental health and social exclusion. The term ”NEET” emerged in the late 1990s in
the United Kingdom and has been used to capture disengagement and social exclusion among young adults
up to the age of 35 in some countries.
The phenomenon of being NEET is multifaceted and influenced by various factors encompassing individ-
ual characteristics, family background, socioeconomic status, educational achievements, aspirations, mental
health, and environmental conditions. As a result, numerous studies have sought to dissect the complex
interplay of these factors and shed light on the predictors of NEET status. Here we review and synthesize a
range of studies that explore the determinants and consequences of being NEET.
The literature surrounding NEET status and its correlates presents a mosaic of findings that underscore the
intricate relationship between various factors and the likelihood of becoming NEET. Studies have illuminated
the role of family socioeconomic status, parental education, and household income as influential factors. For
instance, parental socioeconomic resources, including low education, unemployment, and economic adversity,
have been linked to an increased risk of NEET status. Additionally, adverse childhood experiences, such as
abuse, neglect, parental substance use, and witnessing domestic violence, have been identified as predictors
of NEET status, though their influence is somewhat modest when accounting for socioeconomic status.
Educational attainment emerges as a powerful predictor, with cognitive abilities and aspirations playing
vital roles. Cognitive abilities, as measured by key stage test scores, have shown consistent associations
with the risk of becoming NEET. Aspirations, both of parents and young individuals, hold considerable
sway, influencing the transition from education to employment. Moreover, health status, particularly mental
health, has garnered increased attention as a determinant of NEET status. Recent trends indicate a rising
correlation between self-reported mental ill health and NEET status, with mental health having the largest
effect on the probability of being NEET, especially among males.
The impact of environmental factors cannot be underestimated, as evidenced by the variation in NEET
rates across different regions and local labor market conditions. Early leaving from education, referred to as
”EL,” has emerged as a related concept, demonstrating the need to differentiate between education-related
disengagement and broader social exclusion. The complex interplay of these factors highlights the need for
comprehensive and multifaceted interventions to address the NEET phenomenon effectively.
To sum up, the landscape of NEET research reveals a nuanced web of influences that shape the transition
from education to employment for young people. Individual characteristics, family background, educational
achievements, mental health, and environmental conditions collectively contribute to the risk of being NEET.
Understanding these determinants and their intricate connections is essential for formulating targeted policies
and interventions that can effectively address the challenges faced by NEET individuals. As the research
continues to evolve, there is a growing recognition of the need to consider both cognitive and non-cognitive
factors, socioeconomic resources, aspirations, and mental health in designing strategies that support young
people’s successful transition into adulthood.

</details>

<br>

# Contributors:

**Andrew Kitchen (Project Manager)** 

**Nazeefa Muzammil (Fellow)** 

**Emmanuel Owusu Ahenkan (Fellow)** 

**Mahima Tendulkar (Fellow)** 

**Yannick Stadtfeld (Fellow)** 

**Aygul Zagidullina (Technical Mentor)**

<br>

# Project Partners
Bradford

Solihull

Wolverhampton

Buckinghamshire

EY Foundation

<br>

# Project Resources

Poster: available [here](https://github.com/DSSGxUK/s23_neet/blob/master/documents/NEET%20Poster.pdf) 

Presentation: available [here](https://github.com/DSSGxUK/s23_neet/blob/master/documents/Shard-Final-Presentation.pdf)

Demo at the Streamlit Cloud:[here](https://s23-neet-cloud-dssg-uk.streamlit.app/)

<br>

<details>
<summary>  <h1> Data Description </h1> </summary>

# NCCIS Data

National Client Caseload Information System (NCCIS) data is submitted to the Department
for Education(DfE) by the local authorities. It monitors and records the extent to which the individual is involved with education and training. It is the file which contains the target variable for our prediction model (through the activity codes).

# School Census Data
This data provides demographic information about students such as gender, ethnicity, age, language, eligibility for Free School Meals (FSMs) or Special Educational Needs (SENs). 

# KS4 Data
It holds information related to the student's grades and various attainment scores.

# Attendance Data
This data captures the attendance of students along with features as termly sessions, absences, and reasons for absences, e.g. exclusions, late entries etc. 

# School Level Data

The data is obtained 
from https://www.find-school-performance-data.service.gov.uk/download-data. This source con-
tains the school performance data for all of the England, and it was filtered at the local authority basis. This school performance data included information about the
school postcode, that was used for the feature engineering to calculate the distance from the individual’s place of living
to the school where they study. In addition to this, the categorisation of schools based on the Ofsted ratings helped us distinguish the relative performance of the school.

# Socio-Economic Factors

The data set called as English Indices of
Deprivation https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019 was used to source multiple scores. It
is recorded every four years. The latest one recorded until now is for the year 2019. It provides
the information about Income Deprivation Affecting Children Index (IDACI) and other scores, that help to
categorise the living area of a student according to the various bands.

</details>


<br>

# Potential Risk Factors
* GCSE Attainment
* Absences
* Support Level
* Free School Meals
* Special Education Needs
* IDACI score of the living area
* School-to-home distance

<br>

# Installation of the Package (neet)

To install the "neet" package at your local machine, download or clone the GitHub repository, later once opened in the IDE, run the following commands in the terminal:

```bash
pip install -r requirements.txt 

pyenv virtualenv 3.10.6 neetenv # create virtual environment called neetenv based on python version 3.10.6
pyenv local neetenv # set neetenv as local virtual environment

pip install -e . 
```

# Running the App (Streamlit)

To run the app in the browser (localhost), run the following command in the terminal:

```bash

make run_streamlit

# streamlit run ./neet/streamlit_api/Home.py 
# alternative option

```

# Deployment of the App (Docker)

To build the Docker image and run it use the following commands in the terminal: 

```bash
docker build -t neet-image .

docker run -p 8501:8501 --env-file .env --name neet-container neet-image
```

After that, you will be able to see the result at the local host. 

Visit [http://localhost:8501/](http://localhost:8501/) and check out the running NEETalert App. 

<br>

Additionally, please check the Makefile for the useful docker and other commands. 
