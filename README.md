#begin by importing neccesary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#load the data set using the following command
pd.read_csv("dataset.csv")
#visualize the data set using commands like
dataset.head()  it gives the first five colomns
dataset.describe  gives summary stattistics 
dataset.info
dataset.shape
# Check for missing values
(dataset.isnull().sum())
# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(dataset['price'], ax = axs[0,0])
plt2 = sns.boxplot(dataset['area'], ax = axs[0,1])
plt3 = sns.boxplot(dataset['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(dataset['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(dataset['stories'], ax = axs[1,1])
plt3 = sns.boxplot(datasetdf['parking'], ax = axs[1,2])
# Outlier Treatment
# Price and area have considerable outliers.
# We can drop the outliers as we have sufficient data.
# outlier treatment for price
plt.boxplot(df.price)
Q1 = df.price.quantile(0.25)
Q3 =df.price.quantile(0.75)
IQR = Q3 - Q1
df = df[(df.price >= Q1 - 1.5*IQR) &(df.price <= Q3 + 1.5*IQR)]

****#Visualising Numeric Variables**
**#make a pairplot of all the numeric variables
**Visualising Categorical Variables**
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad', y = 'price', data = dataset)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data = dataset)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data = dataset)
plt.subplot(2,3,4)
datasetplt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y = 'price', data = dataset)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data = dataset)
plt.show()
# List of variables to map

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the df list
df[varlist] = df[varlist].apply(binary_map)

# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'
status = pd.get_dummies(df['furnishingstatus'])

# Check what the dataset 'status' looks like
status.head()

# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(df['furnishingstatus'], drop_first = True)

# Add the results to the original housing dataframe

df = pd.concat([df, status], axis = 1)

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 100)

# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()

**MODEL BUILDING**
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]

# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)
# Making predictions
y_pred = lm.predict(X_test_rfe)
****model evaluation**
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)
