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
# sns.heatmap(df.corr(numeric_only=True),  annot = True)
# plt.show()

# sns.boxplot(data= df, x ="year", y = "price")
# plt.xticks(rotation = 90)
# plt.show()

# sns.scatterplot(data = df, x = "mileage", y = "price")
# plt.show()

# sns.boxplot(data = df, x = "engineSize", y = "price")
# plt.show()

# sns.boxplot(data = df, x = "transmission", y = "price")
# plt.show()


# sns.boxplot(data = df, x = "fuelType", y = "price")
# plt.show()


# sns.boxplot(data = df, x = "model", y = "price")
# plt.xticks(rotation = 90)
# plt.show()

# sns.boxplot(data = df, x = "tax", y = "price")
# plt.xticks(rotation = 90)
# plt.show()

# sns.boxplot(data = df, x = "mpg", y = "price")
# plt.xticks(rotation = 90)
# plt.show()

x = df.drop(columns = ['price'], axis = 1)
y = df['price']

x_one_encoded = pd.get_dummies(x, columns = ['model', 'transmission', 'fuelType'], drop_first = True)

from sklearn.preprocessing import LabelEncoder

columns = ['model', 'transmission', 'fuelType']
Xlable = x.copy()
label_encoders = {}
for col in columns:
    le = LabelEncoder()
    Xlable[col] = le.fit_transform(Xlable[col].astype(str))
    label_encoders[col] = le

from sklearn.preprocessing import StandardScaler
num_columns = ['year', 'mileage', 'engineSize', 'tax', 'mpg']
scaler = StandardScaler()
x_one_encoded[num_columns] = scaler.fit_transform(x_one_encoded[num_columns])

Xlable[['year', 'mileage', 'engineSize', 'tax', 'mpg', 'fuelType', 'transmission', 'model']] = scaler.fit_transform(Xlable[['year', 'mileage', 'engineSize', 'tax', 'mpg', 'fuelType', 'transmission', 'model']])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(x_one_encoded, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R-squared:", r2)



X_train, X_test, y_train, y_test = train_test_split(Xlable, y, test_size = 0.2, random_state = 42)

model2 = LinearRegression()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R-squared:", r2)
