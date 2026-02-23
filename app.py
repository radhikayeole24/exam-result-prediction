# =====================================
#  EXAM RESULT PROJECT
# =====================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# =====================================
# 1. AUTO CREATE DATASET (NO FILE NEEDED)
# =====================================

data = {
    "Student_ID":[1,2,3,4,5,6,7,8,9,10],
    "gender":["Male","Female","Female","Male","Male","Female","Male","Female","Male","Female"],
    "study_hours":[3,1,2,4,1,3,2,1,4,1],
    "attendance":[85,60,78,90,55,72,66,50,88,58],
    "internal_marks":[18,10,15,20,8,14,12,7,19,9],
    "assignment_marks":[20,12,18,19,10,15,14,9,20,11],
    "previous_marks":[75,45,65,80,40,60,55,35,82,42],
    "practical_marks":[22,10,20,25,12,18,15,10,24,12],
    "backlogs":[0,2,0,0,3,1,1,2,0,2],
    "result_status":["Pass","Fail","Pass","Pass","Fail","Pass","Pass","Fail","Pass","Fail"],
    "final_percentage":[78,42,68,88,35,65,58,30,86,40]
}

df = pd.DataFrame(data)

print("Dataset Created Successfully")
print(df.head())

# =====================================
# 2. DATA PREPROCESSING
# =====================================

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features & Targets
X = df.drop(["result_status","final_percentage"], axis=1)
y_class = df["result_status"]      # Pass/Fail
y_reg = df["final_percentage"]     # Percentage

# =====================================
# 3. TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# =====================================
# 4. CLASSIFICATION MODEL (PASS/FAIL)
# =====================================

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_clf.fit(X_train, y_train_c)

y_pred_c = rf_clf.predict(X_test)

print("\n===== RESULT PREDICTION =====")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print(classification_report(y_test_c, y_pred_c))

# =====================================
# 5. REGRESSION MODEL (PERCENTAGE)
# =====================================

rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_reg.fit(X_train_r, y_train_r)

y_pred_r = rf_reg.predict(X_test_r)

print("\n===== PERCENTAGE PREDICTION =====")
print("MAE:", mean_absolute_error(y_test_r, y_pred_r))
print("R2 Score:", r2_score(y_test_r, y_pred_r))

# =====================================
# 6. SAMPLE STUDENT PREDICTION
# =====================================

sample_student = X.iloc[0:1]

result_prediction = rf_clf.predict(sample_student)[0]
percentage_prediction = rf_reg.predict(sample_student)[0]

print("\n===== SAMPLE STUDENT RESULT =====")

if result_prediction == 1:
    print("Predicted Result: PASS")
else:
    print("Predicted Result: FAIL")

print("Predicted Final Percentage:", round(percentage_prediction,2))
