#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: Sat Mar 26 2022


import pandas as pd

df2 = pd.read_csv(
    "JEV_Microarray_23March2022_Final_data/Genes_having_frequency_equal_to_greater_than_8.csv"
)
print(df2.shape)
print(df2.columns)
print(df2.head())
df2_T = df2.T
print(df2_T.shape)
print(df2_T.head())


df2_T.columns = df2_T.iloc[0]
df2_T = df2_T[1:]

print(df2_T.shape)
print(df2_T.head())

print(df2_T.columns)

for i in df2_T.columns:
    # print(i)
    df2_T.loc[(df2_T[i] >= 1.5), i] = 1
    df2_T.loc[(df2_T[i] <= -0.5), i] = -1

print(df2_T.head(11))
df2_T.to_csv("JEV_correlation_binary_inputfile.csv")


def generate_list(y):
    list_result = []
    for i in tqdm(y.columns):
        a = y.loc[y[i].notnull()].index.values
        if len(list_result) == 0:
            if list(a) != []:
                list_result.append(list(a))
        else:
            break
    print(list_result[0])
    return list_result[0]


def generate_result(list_result, k):
    result = []
    for a, b in list(combinations(list_result, k)):
        # print(a,b)

        if a != b:

            l = len([i for i in x[a].to_list() if i != 0])
            m = len([i for i in x[b].to_list() if i != 0])
            n = round((l / 11), 3)
            o = round((m / 11), 3)
            result.extend([a, b, round(y.loc[a, b], 3), l, m, n, o])

    return result


import time
from itertools import combinations
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    t1 = time.time()
    df = pd.read_csv(
        "JEV_Microarray_23March2022_Final_data/JEV_correlation_binary_inputfile.csv"
    )
    x = df.iloc[:, 1:]
    print(x.head())
    y = x.corr()
    print(y.head())
    list_result = generate_list(y)
    result = generate_result(list_result, 2)
    print(result)
    t2 = time.time()

    with open(
        "JEV_Microarray_23March2022_Final_data/JEV_Result_Correlation_Top_42_genes_26march2022.csv",
        "w",
    ) as wf:
        wf.write(
            "gene_1\tgene_2\tCorrelation Value\tFrequency_gene_1\tFrequency_gene_2\tPercentage_gene_1\tPercentage_gene_2\n"
        )
        wf.writelines(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                result[i],
                result[i + 1],
                result[i + 2],
                result[i + 3],
                result[i + 4],
                result[i + 5],
                result[i + 6],
            )
            for i in range(0, len(result), 7)
        )

    print(t2 - t1)
