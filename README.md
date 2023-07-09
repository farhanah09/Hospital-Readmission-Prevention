# Hospital-Readmission-Prevention
This GitHub repository contains the code for an algorithm developed to lower hospital readmission rates by implementing a telehealth system for the right patients. The algorithm aims to provide remote healthcare services to patients, reducing the need for frequent hospital visits and improving patient outcomes. Additionally, the algorithm is designed to save costs for the hospital by optimizing resource allocation and prioritizing high-risk patients for telehealth interventions.

This project was initially developed as part of the Big Data (45980) course at CMU Tepper, where the focus was on leveraging large-scale data analytics techniques to address healthcare challenges. The dataset used in this project was obtained from https://doi.org/10.1155/2014/781670. The dataset includes relevant patient information, medical histories, and readmission records, enabling the algorithm to identify patterns and predict readmission risks.

The code in this repository implements various machine learning and data analysis techniques to identify patients who would benefit from telehealth services. It includes preprocessing steps, feature engineering, model training, and evaluation scripts. The repository also provides documentation, instructions, and examples to facilitate understanding and usage of the algorithm.

## Data Source - 
https://doi.org/10.1155/2014/781670
The data is also split into training and testing CSV files and is available in the dataset folder

## Requirements
To install the requirements for the code, run the following command - 

```
pip install -r requirements.txt
```

## About - 

This was the background of the problem presented in the class- 

A key performance metric for hospitals is the 30-day unplanned readmission rate -- the proportion of patients discharged from the hospital who had an unplanned readmission within 30 days. Programs like the Hospital Readmissions Reduction Program (HRRP) apply penalties (up to a 3% reduction in payments) to underperforming U.S. hospitals -- resulting in withheld payments in excess of $500 million in 2018.

Problem - 
Suppose you are working for a large hospital system in the United States, and are tasked to assess the impact of telehealth interventions on diabetic patients -- with the ultimate goal of reducing the 30-day readmission rate. The intervention will cost approximately $1,200 per patient. The dataset includes over 100,000 hospital discharges of over 70,000 diabetic patients from 130 hospitals across the United States.

All patients were hospital inpatients for 1-14 days and received both lab tests and medications while in the hospital. The 130 hospitals represented in the dataset vary in size and location.

The dataset is randomly split into train <readmission_train.csv> and test <readmission_test.csv> 
• sets.readmission: 1 if the patient had an unplanned readmission within 30 days, 0 otherwise.
• Patient characteristics: age captures demographic information.
• Recent medical system use: The variables numberEmergency, and numberInpatient capture the number of times the patient used the medical system in the last year.
• Diabetic treatments: A number of variables capture the patient's diabetic treatments: insulin, metformin
• Admission information: numberDiagnoses captures the number of diagnoses the patient had recorded for their admission. There are also a number of variables that indicate whether a patient was diagnosed with various conditions when admitted: diagAnemia, diagAsthma, diagAthlerosclerosis, diagCellulitis, diagCKD, diagDyspnea, diagHeartFailure, diagHypertension, diagHypertensiveCKD, diagPneumonia, and diagSkinUlcer.
• Treatment information: timeInHospital is the number of days the patient was in the hospital, and numNonLabProcedures, and numMedications capture the amount of care the patient received in the hospital.

The cost of admission is about $35,000 and from published information at a similar institution, you estimate that telehealth interventions will reduce the incidence of 30-day unplanned readmissions in the treated population by 25%.