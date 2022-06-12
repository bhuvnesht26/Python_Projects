#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, make_scorer
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


t1 = time.time()
data = pd.read_csv("inputfile_for_ML_fivemer.csv", header=None, index_col=0)
# print(data)
x = data.iloc[:, :32]
# print(x)
y = data.iloc[:, 32]
# print(y)
x = x.div(x.sum(axis=1), axis=0)  # divide each row by sum of row
x = x.to_numpy()

y = y[:, np.newaxis]


knn = KNeighborsClassifier(n_neighbors=5)

t1 = time.time()

scoring = {
    "accuracy": "accuracy",
    "f1_weighted": "f1_weighted",
    "mcc_scorer": make_scorer(matthews_corrcoef),
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}

from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.1)
cv_precision = np.zeros(10)
cv_recall = np.zeros(10)
cv_roc_auc = np.zeros(10)
cv_mcc = np.zeros(10)
cv_accuracy = np.zeros(10)
cv_f1_weighted = np.zeros(10)
for i in range(0, 10):
    cv_score = cross_validate(knn, x, y, scoring=scoring, cv=cv)
    cv_precision[i] = np.mean(cv_score["test_precision"])
    cv_recall[i] = np.mean(cv_score["test_recall"])
    cv_roc_auc[i] = np.mean(cv_score["test_roc_auc"])
    cv_mcc[i] = np.mean(cv_score["test_mcc_scorer"])
    cv_accuracy[i] = np.mean(cv_score["test_accuracy"])
    cv_f1_weighted[i] = np.mean(cv_score["test_f1_weighted"])


t2 = time.time()


print(np.mean(cv_accuracy))
print(np.mean(cv_f1_weighted))
print(np.mean(cv_precision))
print(np.mean(cv_recall))
print(np.mean(cv_roc_auc))
print(np.mean(cv_mcc))
print(f"It took {t2 - t1} seconds to process.")
