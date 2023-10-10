# Predicting students at risk of becoming NEET (Not in Education, Employment or Training)

Welcome to the code repository for the project conducted under Data Science for Social Good - UK 2023 (DSSGx UK).
This README provides an overview of the project, the contributors, the folder structure of the repository, assets, information and guidance from the completed project.

This project aimed to develop predictive models to identify individuals at risk of becoming NEET (Not
in Education, Employment, or Training), for local authorities in England for young people between 16
and 18 years old, and provide an open source tool to visualise the results at a council, school and individual level of granularity. 
The significance of this work was to help prevent educational disengagement, unemployment and safeguarding of mental well-being in vulnerable individuals.
Through the timely implementation of tailored interventions using intelligence provided by the project, we sought to empower the lives of the young people
by ensuring they stayed engaged in education, training or found gainful employment opportunities, rather than become NEET.

<br>

# Contributors:
The following University of Warwick staff and fellows contributed to the project:

**Andrew Kitchen (Project Manager)** 

**Aygul Zagidullina (Technical Mentor)**

**Nazeefa Muzammil (Fellow)** 

**Emmanuel Owusu Ahenkan (Fellow)** 

**Mahima Tendulkar (Fellow)** 

**Yannick Stadtfeld (Fellow)** 

<br>

# Project Partners
The project worked in collaboration with the following partners:

EY Foundation

City of Bradford Metropolitan District Council

City of Wolverhampton Council

Solihull Metropolitan Borough Council

Buckinghamshire Council

<br>

# Background research summary on the implications of becoming NEET

The transition from adolescence to adulthood is a critical phase that shapes an individual’s future prospects,
impacting their education, employment, and overall well-being. Among the challenges that young people
face during this transition is the risk of being categorized as ”NEET” – Not in Education, Employment, or
Training. The NEET status has garnered significant attention due to its association with adverse outcomes,
particularly in terms of mental health and social exclusion. The term ”NEET” emerged in the late 1990s, in
the United Kingdom, and has been used to capture disengagement and social exclusion among young adults
up to the age of 35 in some countries.

<br>

The phenomenon of being NEET is multifaceted and influenced by various factors encompassing individual characteristics, 
family background, socioeconomic status, educational achievements, aspirations, mental
health, and environmental conditions. As a result, numerous studies have sought to dissect the complex
interplay of these factors and shed light on the predictors of NEET status. Here we review and synthesise a
range of studies that explore the determinants and consequences of being NEET.

<br>

The literature surrounding NEET status and its correlations present a mosaic of findings that underscore the
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

<br>

The impact of environmental factors cannot be underestimated, as evidenced by the variation in NEET
rates across different regions and local labor market conditions. Early leaving from education, referred to as
”EL,” has emerged as a related concept, demonstrating the need to differentiate between education-related
disengagement and broader social exclusion. The complex interplay of these factors highlights the need for
comprehensive and multifaceted interventions to address the NEET phenomenon effectively.

<br>

To sum up, the landscape of NEET research reveals a nuanced web of influences that shape the transition
from education to employment for young people. Individual characteristics, family background, educational
achievements, mental health, and environmental conditions collectively contribute to the risk of being NEET.
Understanding these determinants and their intricate connections is essential for formulating targeted policies
and interventions that can effectively address the challenges faced by NEET individuals. As the research
continues to evolve, there is a growing recognition of the need to consider both cognitive and non-cognitive
factors, socioeconomic resources, aspirations, and mental health in designing strategies that support young
people’s successful transition into adulthood.


<br>

# Data Description

## NCCIS Data

National Client Caseload Information System (NCCIS) data is submitted to the Department
for Education(DfE) by the local authorities. It monitors and records the extent to which the individual is involved with education and training. It is the file which contains the target variable for our prediction model (through the activity codes).

## School Census Data

This data provides demographic information about students such as gender, ethnicity, age, language, eligibility for Free School Meals (FSMs) or Special Educational Needs (SENs). 

## KS4 Data

It holds information related to the student's grades and various attainment scores.

## Attendance Data

This data captures the attendance of students along with features as termly sessions, absences, and reasons for absences, e.g. exclusions, late entries etc. 

## Exclusions Data

This data captures the information about an individual’s historical exclusion status.

## School Level Data

The data is obtained 
from https://www.find-school-performance-data.service.gov.uk/download-data. The school performance dataset contains data for all schools in England, and it was filtered at the local authority level. The data includes information about the
school postcode, which was used during feature engineering to calculate the distance from the individual’s place of living to the school where they study. In addition to this, the categorisation of schools based on the Ofsted ratings helped distinguish the relative performance of the school.

## Socio-Economic Factors

The dataset is called the English Indices of
Deprivation and is obtained from https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019. It
is recorded every four years - the latest is for the year 2019. It provides information about Income Deprivation Affecting Children Index (IDACI) and other scores, which help to
categorise the living area of an individual according to various bands of deprivation.

<br>

# Potential Risk Of NEET Indicators (RONI)

Our work concluded the following were the most prevalent RONIs using the range of datasets incorporated within the modelling undertaken:
* GCSE Attainment
* Absences
* Support Level
* Free School Meals
* Special Education Needs
* IDACI score of the living area
* School-to-home distance

<br>

# Project Resources

Video from the Shard: available [here](https://warwick.ac.uk/research/data-science/warwick-data/dssgx/neet_2023.mp4)

Presentation: available [here](https://github.com/DSSGxUK/s23_neet/blob/master/documents/Shard-Final-Presentation.pdf)

Poster: available [here](https://github.com/DSSGxUK/s23_neet/blob/master/documents/NEET%20Poster.pdf) 

NEETalert Tool Demo on the Streamlit Cloud: available [here](https://s23-neet-cloud-dssg-uk.streamlit.app/)

<br>

# Installation

There are two ways to install the NEETalert tool - EITHER manually install the software OR install a pre-configured Docker image.

## 1) Manual installation
There are two stages in this process, as noted below:

### Install pre-requisite software and neet package

To install the "neet" package at your local machine, download or clone the GitHub repository, later once opened in the IDE, run the following commands in the terminal:

```bash
pip install -r requirements.txt 

pyenv virtualenv 3.10.6 neetenv # create virtual environment called neetenv based on python version 3.10.6
pyenv local neetenv # set neetenv as local virtual environment

pip install -e . 
```

### Run the app in Streamlit

To run the app in the browser (localhost), run the following command in the terminal:

```bash

make run_streamlit

# streamlit run ./neet/streamlit_api/Home.py 
# alternative option

```
Once the above has completed, visit [http://localhost:8501/](http://localhost:8501/) and check out the running NEETalert tool. 

## 2) Docker based deployment

To use this method you must have the ability to run the open source Docker Engine but you do not need Docker Desktop.

Build the Docker image and run it using the following commands in the terminal: 

```bash
docker build -t neet-image .

docker run -p 8501:8501 --env-file .env --name neet-container neet-image
```

Once the above has completed, visit [http://localhost:8501/](http://localhost:8501/) and check out the running NEETalert tool.

<br>

Additionally, please check the Makefile for docker and other useful commands. 
