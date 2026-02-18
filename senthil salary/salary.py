import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Salary Data.csv")

df.head()

df.shape
df.info()
df.describe()

df.isnull().sum()

df["Age"].fillna(df["Age"].mean())
df["Years of Experience"].fillna(df["Years of Experience"].mean())
df["Gender"].fillna(df["Gender"].mode()[0])
df["Education Level"].fillna(df["Education Level"].mode()[0])
df["Job Title"].fillna(df["Job Title"].mode()[0])
df = df.dropna(subset=["Salary"])

df.isnull().sum()

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['Education Level'] = le.fit_transform(df['Education Level'])
df['Job Title'] = le.fit_transform(df['Job Title'])

df.head()
plt.figure(figsize=(7,4))
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(df['Salary'], kde=True)
plt.title("Salary Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(y=df['Salary'])
plt.title("Salary Boxplot")
plt.show()

plt.figure(figsize=(7,4))
sns.scatterplot(x='Age', y='Salary', data=df)
plt.title("Age vs Salary")
plt.show()

plt.figure(figsize=(7,4))
sns.scatterplot(x='Years of Experience', y='Salary', data=df)
plt.title("Experience vs Salary")
plt.show()

plt.figure(figsize=(7,4))
sns.boxplot(x='Gender', y='Salary', data=df)
plt.title("Gender vs Salary")
plt.show()

plt.figure(figsize=(7,4))
sns.boxplot(x='Education Level', y='Salary', data=df)
plt.title("Education vs Salary")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['Age','Years of Experience','Salary','Gender','Education Level']])
plt.show()

X = df[['Age','Gender','Education Level','Years of Experience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²  :", r2_score(y_test, y_pred))

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Age'], df['Years of Experience'], df['Salary'])

ax.set_xlabel("Age")
ax.set_ylabel("Years of Experience")
ax.set_zlabel("Salary")
ax.set_title("3D Plot: Age vs Experience vs Salary")

plt.show()

pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])