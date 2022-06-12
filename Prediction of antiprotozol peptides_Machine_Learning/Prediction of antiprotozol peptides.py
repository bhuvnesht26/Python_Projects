#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: Wed Dec 29 2021


import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import math
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


with open(
    "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Training_data_index.txt"
) as rf:
    data = rf.readlines()
print(len(data))
data = [int(i.rstrip("\n")) for i in data]
data


with open(
    "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Testing_data_index.txt"
) as rf:
    data_val = rf.readlines()
print(len(data_val))
data_val = [int(i.rstrip("\n")) for i in data_val]
data_val

df = pd.read_csv(
    "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/22_features_using_C_0.003.csv",
    header=None,
)
print(df.head())
cols = df.shape[1]
print(cols)

a = df.iloc[:, : cols - 1]
print(a.head())

b = df.iloc[:, cols - 1]
print(b.head())
print(b.tail())

# x_train, x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,stratify = y)
a_train = a.iloc[data, :]
a_val = a.iloc[data_val, :]
b_train = b.iloc[data]
b_val = b.iloc[data_val]


################### XgBoost ######################3

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    random_state=10,
    max_depth=30,
    reg_lambda=30,
)
model.fit(a_train, b_train)
b_pred = model.predict(a_val)

predictions = [round(value) for value in b_pred]

accuracy = accuracy_score(b_val, b_pred)

accuracy

from sklearn.metrics import roc_auc_score

y_predict = model.predict(a_val)
y_predict_proba = model.predict_proba(a_val)

list_1_val = b_val.tolist()
list_2 = y_predict_proba.tolist()
list_3_prob = []
for i in list_2:
    list_3_prob.append(i[1])

auc_score1 = roc_auc_score(b_val, y_predict_proba[:, 1])
print(auc_score1)


def probability(list_1_val, list_3_prob, theshold):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(list_1_val)):
        if list_1_val[i] == 1 and list_3_prob[i] >= theshold:
            TP += 1
        elif list_1_val[i] == 0 and list_3_prob[i] <= theshold:
            TN += 1
        elif list_1_val[i] == 0 and list_3_prob[i] > theshold:
            FP += 1
        elif list_1_val[i] == 1 and list_3_prob[i] < theshold:

            FN += 1

    Sensitivity = (TP / (TP + FN)) * 100
    Specificity = (TN / (TN + FP)) * 100
    Accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    MCC = ((TP * TN) - (FP * FN)) / (
        math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    )
    return TP, TN, FP, FN, Sensitivity, Specificity, Accuracy, MCC


(
    True_Positive,
    True_Negative,
    False_Positive,
    False_Negative,
    Sensitivity,
    Specificity,
    Accuracy,
    MCC,
) = probability(list_1_val, list_3_prob, 0.51)
print(
    True_Positive,
    True_Negative,
    False_Positive,
    False_Negative,
    Sensitivity,
    Specificity,
    Accuracy,
    MCC,
)


############################ saving model#############


#######################1.XgBoost############################
model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    random_state=10,
    max_depth=30,
    reg_lambda=30,
)
model.fit(a_train, b_train)
filename = "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Models/XgBoost_model.sav"
joblib.dump(model, filename)


####################2.Logistic Regression ################
model = LogisticRegression(solver="liblinear", penalty="l1", C=10, random_state=42)
model.fit(a_train, b_train)
filename = "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Models/Logistic_Regression_model.sav"
joblib.dump(model, filename)


####################3.Decision tree #####################3

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    min_samples_split=20,
    max_features="auto",
    random_state=42,
)
model.fit(a_train, b_train)
filename = "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Models/Decision_Tree_model.sav"
joblib.dump(model, filename)


#######################4.Random Forest####################3

model = RandomForestClassifier(
    n_estimators=400,
    criterion="entropy",
    max_depth=10,
    min_samples_split=20,
    max_features="auto",
    random_state=42,
)
model.fit(a_train, b_train)
filename = "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Models/Random_Forest_model.sav"
joblib.dump(model, filename)


########################5. SVM #############################

model = SVC(kernel="poly", degree=3, C=1000, probability=True)
model.fit(a_train, b_train)

filename = "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Models/SVM_model.sav"
joblib.dump(model, filename)


###################again checking ###############

loaded_model = joblib.load(
    "28Dec2021_Final_Antiprotozoal_Work/SVCL1_feature_Selection_Training_Testing_Datasets/Antibacterial_feature_selection_training_testing_data/Models/XgBoost_model.sav"
)


y_predict = loaded_model.predict(a_val)
y_predict_proba = loaded_model.predict_proba(a_val)


list_1_val = b_val.tolist()
list_2 = y_predict_proba.tolist()
list_3_prob = []
for i in list_2:
    list_3_prob.append(i[1])

auc_score1 = roc_auc_score(b_val, y_predict_proba[:, 1])
print(auc_score1)


def probability(list_1_val, list_3_prob, theshold):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(list_1_val)):
        if list_1_val[i] == 1 and list_3_prob[i] >= theshold:
            TP += 1
        elif list_1_val[i] == 0 and list_3_prob[i] <= theshold:
            TN += 1
        elif list_1_val[i] == 0 and list_3_prob[i] > theshold:
            FP += 1
        elif list_1_val[i] == 1 and list_3_prob[i] < theshold:

            FN += 1

    Sensitivity = (TP / (TP + FN)) * 100
    Specificity = (TN / (TN + FP)) * 100
    Accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    MCC = ((TP * TN) - (FP * FN)) / (
        math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    )
    return TP, TN, FP, FN, Sensitivity, Specificity, Accuracy, MCC


(
    True_Positive,
    True_Negative,
    False_Positive,
    False_Negative,
    Sensitivity,
    Specificity,
    Accuracy,
    MCC,
) = probability(list_1_val, list_3_prob, 0.51)
print(
    True_Positive,
    True_Negative,
    False_Positive,
    False_Negative,
    Sensitivity,
    Specificity,
    Accuracy,
    MCC,
)
