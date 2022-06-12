#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: Tue, Jan 4, 2022


from sklearn.preprocessing import StandardScaler


import pandas as pd

df = pd.read_excel("MRMR_SVC_Features_1Jan2022.xlsx", sheet_name="Antiviral_MRMR")
print(df)

Features_MRMR = df["Features"].to_list()
print(Features_MRMR)
print(len(Features_MRMR))

#################### Positive ####################

df1 = pd.read_csv(
    "Individual_Pfeatures/Pfeature_Antiprotozoal_Positive_Features_with_header.csv"
)
print(df1.head())


df3_Positive = df1.loc[:, Features_MRMR]
print(df3_Positive.shape)
print(df3_Positive.head())
df3_Positive.to_excel(
    "MRMR_Boxplot_4Jan2022/Inputfiles/Antiviral/Positive_MRMRAntiprotozoal.xlsx"
)

df2 = pd.read_csv(
    "Individual_Pfeatures/Pfeature_Antiviral_Negative_Features_with_header.csv"
)
print(df2.head())

df4_Negative = df2.loc[:, Features_MRMR]
print(df4_Negative.shape)
print(df4_Negative.head())
df4_Negative.to_excel(
    "MRMR_Boxplot_4Jan2022/Inputfiles/Antiviral/Negative_MRMRAntiviral.xlsx"
)


from sklearn.preprocessing import StandardScaler


import pandas as pd

df = pd.read_excel("MRMR_SVC_Features_1Jan2022.xlsx", sheet_name="Antiviral_MRMR")
print(df)

Features_MRMR = df["Features"].to_list()
print(Features_MRMR)
print(len(Features_MRMR))

#################### Positive ####################

df1 = pd.read_csv(
    "Individual_Pfeatures/Pfeature_Antiprotozoal_Positive_Features_with_header.csv"
)
print(df1.head())


df3_Positive = df1.loc[:, Features_MRMR]
print(df3_Positive.shape)
print(df3_Positive.head())
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(df3_Positive)
print(scaled)
scaled = pd.DataFrame(scaled)
scaled.to_excel(
    "MRMR_Boxplot_4Jan2022/Inputfiles/Antiviral/Positive_MRMR_Antiprotozoal_Scaled.xlsx"
)


df_combined_Positive = pd.DataFrame()
df_P = pd.read_excel(
    "MRMR_Boxplot_4Jan2022/Inputfiles/Antiviral/Positive_MRMR_Antiprotozoal_Scaled.xlsx",
    header=None,
)
print(df_P.head())
cols = df_P.shape[1]
cols

for i in range(0, cols, 2):
    df_col = df_P.iloc[:, i : i + 2]
    print(i, i + 2)
    print(df_col.head())
    df_col.columns = ["Value", "Feature"]
    df_combined_Positive = pd.concat([df_combined_Positive, df_col], axis=0)
print(len(df_combined_Positive))
df_combined_Positive.reset_index(drop="index", inplace=True)

df_combined_Positive["Data"] = "Positive"

print(df_combined_Positive)
print(df_combined_Positive.shape)


#################### Negative ###########

df2 = pd.read_csv(
    "Individual_Pfeatures/Pfeature_Antiviral_Negative_Features_with_header.csv"
)
print(df2.head())

df4_Negative = df2.loc[:, Features_MRMR]
print(df4_Negative.shape)
print(df4_Negative.head())
scaled = scaler.fit_transform(df4_Negative)
print(scaled)
scaled = pd.DataFrame(scaled)
scaled.to_excel(
    "MRMR_Boxplot_4Jan2022/Inputfiles/Antiviral/Negative_MRMR_Antiviral_Scaled.xlsx"
)


df_combined_Negative = pd.DataFrame()
df_N = pd.read_excel(
    "MRMR_Boxplot_4Jan2022/Inputfiles/Antiviral/Negative_MRMR_Antiviral_Scaled.xlsx",
    header=None,
)
print(df_N.head())
cols = df_N.shape[1]
cols

for i in range(0, cols, 2):
    df_col = df_N.iloc[:, i : i + 2]
    print(i, i + 2)
    print(df_col.head())
    df_col.columns = ["Value", "Feature"]
    df_combined_Negative = pd.concat([df_combined_Negative, df_col], axis=0)
print(len(df_combined_Negative))
df_combined_Negative.reset_index(drop="index", inplace=True)

df_combined_Negative["Data"] = "Negative"

print(df_combined_Negative)
print(df_combined_Negative.shape)


#################concatination ##############


df_Positive_Negative = pd.concat([df_combined_Positive, df_combined_Negative], axis=0)
df_Positive_Negative.reset_index(drop="index", inplace=True)

print(df_Positive_Negative.shape)


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(30, 15))

ax = sns.boxplot(x="Feature", y="Value", hue="Data", data=df_Positive_Negative)
# ax.tick_params( top=False, bottom= True,labeltop=True,labelbottom = False,left = True)
plt.xticks(size=15, fontfamily="arial", fontweight="bold", rotation=90)
plt.yticks(size=15, fontfamily="arial", fontweight="bold")
plt.xlabel("Features", fontsize=30)
plt.ylabel("Values", fontsize=30)
plt.legend(loc="best", fontsize="xx-large")
plt.savefig("MRMR_Boxplot_Antiviral.eps", pad_inches=0.1, bbox_inches="tight")
