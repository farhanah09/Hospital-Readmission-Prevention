import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sympy import *

train_data = pd.read_csv("Dataset/readmission_train.csv")
test_data = pd.read_csv("Dataset/readmission_test.csv")

# train and test split
y_train = train_data["readmission"]
X_train = train_data.drop("readmission", axis=1)
y_test = test_data["readmission"]
X_test = test_data.drop("readmission", axis=1)

# checking the correlation between the elements

a = pd.DataFrame(X_train.corr())
print(a)

# finding values that are highly correlated to remove them from the model to avoid multicollinearity
min_age = 0.4
max_age = 1.0
filtered_df = a[(a >= min_age) & (a <= max_age)]

print(filtered_df)
# Logistic Regression Model

# the elements timeInHospital and numMedications seem to be highly colinear, hence will be removed from the model
logreg_mod_sm = smf.logit(
    data=train_data,
    formula="readmission ~ age+numberEmergency+numberInpatient+insulin+metformin+numberDiagnoses+diagAnemia+diagAsthma+diagAthlerosclerosis+diagCellulitis+diagCKD+diagDyspnea+diagHeartFailure+diagHypertension+diagHypertensiveCKD+diagPneumonia+diagSkinUlcer+numNonLabProcedures",
)

logreg_sm = logreg_mod_sm.fit()

logreg_sm.summary()

logreg_mod_sm = smf.logit(
    data=train_data,
    formula="readmission ~ age+numberEmergency+numberInpatient+insulin+numberDiagnoses+diagAsthma+diagAthlerosclerosis+diagCellulitis+diagCKD+diagDyspnea+diagHeartFailure+diagHypertension+diagHypertensiveCKD+diagPneumonia+diagSkinUlcer+numNonLabProcedures",
)
logreg_sm = logreg_mod_sm.fit()
logreg_sm.summary()

# Probabilities of admission

p = Symbol("p")
tele_admitted = 0.75 * (p)
tele_notadmitted = 1 - tele_admitted
admitted = p
not_admitted = 1 - admitted

# costs of admission

cost_admission = 35000
cost_telehealth = 1200
cost_noadmission = 0

# calculating the probability of admisssion -

x = (
    cost_admission * p
    + (1 - p) * cost_noadmission
    - tele_admitted * (cost_admission + cost_telehealth)
    - (1 - tele_admitted) * (cost_telehealth)
)
my_threshold = solve(x, p)
my_threshold = my_threshold[0]
print(my_threshold)

pred_prob_logreg_sm = logreg_sm.predict(X_test)
class_logreg_sm = (pred_prob_logreg_sm > my_threshold).astype(int)
cm_logreg_sm = confusion_matrix(y_test, class_logreg_sm)

cm_logreg_sm  # The confusion matrix

TN = cm_logreg_sm[
    0, 0
]  # number of people who weren't provided with telehealth and didn't get admitted
TP = cm_logreg_sm[
    1, 1
]  # number of people who were provided with telehealth and did get admitted
FN = cm_logreg_sm[
    1, 0
]  # number of people who were provided with telehealth and didn't get admitted
FP = cm_logreg_sm[
    0, 1
]  # number of people who weren't provided with telehealth and did get admitted

accuracy = (cm_logreg_sm[0, 0] + cm_logreg_sm[1, 1]) / sum(sum(cm_logreg_sm))
sensitivity = (cm_logreg_sm[1, 1]) / (cm_logreg_sm[1, 0] + cm_logreg_sm[1, 1])
specificity = (cm_logreg_sm[0, 0]) / (cm_logreg_sm[0, 0] + cm_logreg_sm[0, 1])
print("Accuracy: ", round(accuracy, 2))
print("Sensitivity: ", round(sensitivity, 2))
print("Specificity: ", round(specificity, 2))

x = (
    (FP * cost_telehealth)
    + (FN * cost_admission)
    + (TP * 0.75) * (cost_admission + cost_telehealth)
)  # this multiplies the
# confusion matrix elements and their respective costs. The people who were admitted into the hospital while using telehealth
# reduces by 25% as per the question. Hence, multiplied by 0.75
print("The cost of admission is $", x)
print("The cost of admission per patient is $", x / 30530)

counts = test_data[
    "readmission"
].value_counts()  # counting the people who were readmitted into the hospital
y = counts[1] * 35000  # this is the cost of admission
print("The cost of admission is $", y)
print("The cost of admission per patient is $", y / 30530)

print("The profit of using telehealth is $", ((y - x) / 30530))