#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: Sat Apr 2, 2022


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel(
    "JEV_Microarray_23March2022_Final_data/Pearson Correlation_113genes_250322_rr.xlsx",
    sheet_name="Heatmap",
)
print(df.head())
gene1 = df["gene_1"].to_list()
print(len(gene1))
gene1_unique = list(set(gene1))
print(len(gene1_unique))

gene2 = df["gene_2"].to_list()
print(len(gene2))
gene2_unique = list(set(gene2))
print(len(gene2_unique))


print(df.columns)

df2 = df.pivot("gene_1", "gene_2", "Correlation Value")
df2 = df2.fillna(0)
print(df2.head())


fig, ax = plt.subplots(figsize=(15, 10))
color_map = plt.cm.get_cmap("YlGnBu_r")
reversed_color_map = color_map.reversed()
sns.set(font_scale=1.2)


sns.heatmap(
    df2,
    linewidths=0.5,
    cmap=reversed_color_map,
    cbar_kws={
        "shrink": 0.70,
    },
    yticklabels=1,
)
ax.tick_params(bottom=True, labelbottom=True, left=True)
plt.xticks(size=12, fontfamily="arial", fontweight="bold", rotation=90)
plt.yticks(size=12, fontfamily="arial", fontweight="bold")
plt.savefig("JEV.eps", pad_inches=0.1, bbox_inches="tight")
