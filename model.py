import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#ANALYSING THE DATA
df = pd.read_csv(r"ford.csv")
print(df.shape)
print(df.info())
print(df.describe()) #gets mean and everything of numeric columns
print(df.isnull().sum())


#EDA FOR PRICE
# sns.histplot(df["price"], bins = 50, kde = True)# kde = true means it shows the line like how my data is distributed
# plt.show()

print(df.corr(numeric_only=True))
sns.heatmap(df.corr(numeric_only=True),  annot = True)
plt.show()
