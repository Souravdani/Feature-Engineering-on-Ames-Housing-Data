# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 18:43:23 2022
AMES HOUSING DATA
@author: Soura
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("G:\\Python\\ML and Data Science\\UNZIP_FOR_NOTEBOOKS_FINAL\\DATA\\Ames_Housing_Data.csv")
df.corr()["SalePrice"].sort_values()



################ DEALING WITH OUTLIERS ##################




sns.scatterplot(data= df, x= "Overall Qual", y= "SalePrice")
sns.scatterplot(data= df, x= "Gr Liv Area", y= "SalePrice")
## We can clearly see some outliers in second scatterplot 
df[(df["Overall Qual"]>8) & (df["SalePrice"]<200000)]
## Intuitive outliers on two features
## Remove three outliers not following trend

drop_ind= df[(df["Gr Liv Area"]>4000) & (df["SalePrice"]<400000)].index
df= df.drop(drop_ind, axis= 0)
sns.scatterplot(data= df, x= "Gr Liv Area", y= "SalePrice")
## Now scatterplot making more sense

## Outliers removed, now saving our data
df= pd.to_csv("Ames_hous_data_ouliers_removed")




############## DEALING WITH MISSING DATA #################



## Description of features of dataset
with open('G:\\Python\\ML and Data Science\\UNZIP_FOR_NOTEBOOKS_FINAL\\DATA\\Ames_Housing_Feature_Description.txt','r') as f: 
    print(f.read())

df= pd.read_csv("G:\\Python\\ML and Data Science\\UNZIP_FOR_NOTEBOOKS_FINAL\\DATA\\Ames_outliers_removed.csv")

# Now, dropping PID or we can also make it our index
df= df.drop("PID", axis=1)
len(df.columns)

df.isnull()  # Boolean value at each data point
df.isnull().sum()

## Better to see the ratio or percentage of missing data
100*df.isnull().sum()/len(df)


## Function for a dataset to get percent nan values
def percent_missing(df):
    percent_nan= 100*df.isnull().sum()/len(df)
    percent_nan= percent_nan[percent_nan>0].sort_values()
    
    return percent_nan

percent_nan= percent_missing(df)
## We have got the percentage data missing for each of our columns
## Now we can do a barplot to visualize it

sns.barplot(x= percent_nan.index, y= percent_nan)
plt.xticks(rotation=90);

## We can see some of columns have very high percent of null data
## But for features having null values less than 1 percent, it is feasible here to drop
## because we have good number od data rows

percent_nan[percent_nan<1]
df[df["Electrical"].isnull()]
df[df["Garage Area"].isnull()]
df[df["Bsmt Half Bath"].isnull()]
## Different rows 

df= df.dropna(axis= 0, subset= ['Electrical','Garage Cars'])
percent_nan= percent_missing(df)
percent_nan[percent_nan<1]  ## Some common feature nan rows has been dropped

df[df["Bsmt Half Bath"].isnull()] # Col- 1341,1497
df[df["Bsmt Full Bath"].isnull()] # Col- 1341,1497

df[df["Bsmt Unf SF"].isnull()] # Col- 1341

## From domain knowledge we got to know that basement features na can be 
# filled with zero value

## BSMT NUMERIC COLS --> fill na
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

## BSMT STRING COLS -->
bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')

percent_nan = percent_missing(df)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);
## Now we have very less features having na values

## Based on the Description Text File, Mas Vnr Type and Mas Vnr Area being 
# missing (NaN) is likely to mean the house simply just doesn't have a 
# masonry veneer, in which case, we will fill in this data as we did before.

df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)
percent_nan = percent_missing(df)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


## Now focusing on other columns having na beyond one percent threshold set by us:
    
## Filling the na values if na values are in substantial proportion of our 
# dataset is a tricky work and statistical knowledge comes handy
## We can also use other features data to predict na for a feature column

gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None') 

percent_nan = percent_missing(df)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);
 
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)

df['Fireplace Qu'] = df['Fireplace Qu'].fillna("None")

## Only Lot Frontage missing -- Numerical data
# Statistical estimation can be done
## Since Lot frontage is the linear feet of street connected to property

plt.figure(figsize=(8,12))
sns.boxplot(x='Lot Frontage',y='Neighborhood',data=df,orient='h')

## Its reasonable to predict na values from the Neighborhood feature
df.groupby('Neighborhood')['Lot Frontage']
df.groupby('Neighborhood')['Lot Frontage'].mean()

## Pandas transformation: call group by and fill based on that
df.iloc[21:26]['Lot Frontage']
df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))
df['Lot Frontage'] = df['Lot Frontage'].fillna(0)

percent_nan = percent_missing(df)
percent_nan

## Missing values have ben dealt with




#################### DEALING WITH CATEGORIES ####################



df.isnull().sum()
with open('../DATA/Ames_Housing_Feature_Description.txt','r') as f: 
    print(f.read())

# Convert to String
df['MS SubClass'] = df['MS SubClass'].apply(str)

## Tells the datatypes of our features
df.select_dtypes(include='object')

## Splitting our dataframe into two parts Numeric features and string features
df_nums = df.select_dtypes(exclude='object')
df_objs = df.select_dtypes(include='object')

df_objs = pd.get_dummies(df_objs,drop_first=True)
# Now we have created dumy vars, we will concatenate the splitted df

final_df = pd.concat([df_nums,df_objs],axis=1)
final_df
final_df.to_csv('C:\\Users\\Soura\\Downloads\\DATA\\AMES_Final_DF.csv')





############# MODELLING OUR DATA (LINEAR MODEL) ############




df= pd.read_csv("C:\\Users\\Soura\\Downloads\\DATA\\AMES_Final_DF.csv")
df.info()
df.columns

X= df.drop("SalePrice", axis=1)
y= df["SalePrice"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

## Scaling the data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

scaled_X_train= scaler.fit_transform(X_train)
scaled_X_test= scaler.transform(X_test)



################### ELASTIC NET #################

## Model creation
from sklearn.linear_model import ElasticNet
basic_model= ElasticNet()

## We are setting a gridseach cv for optimal values of hyperparameters
## param grid for alpha and l1_ratio
param_grid= {'alpha':[0.1,1,5,10,50,100], 'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}

from sklearn.model_selection import GridSearchCV
grid_model= GridSearchCV(estimator= basic_model,param_grid= param_grid,scoring='neg_mean_squared_error',cv= 5, verbose=1)
grid_model.fit(scaled_X_train, y_train)
grid_model.best_params_  # 'alpha': 100, 'l1_ratio': 1
grid_model
## On training data
y_pred_train= grid_model.predict(scaled_X_train)
y_pred_train

## On test data
y_pred= grid_model.predict(scaled_X_test)
y_pred

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error(y_pred_train, y_train)                       # 13433
np.sqrt(mean_squared_error(y_pred_train,y_train))                # 19745
r2_score(y_pred_train,y_train)                                   # 0.93

mean_absolute_error(y_pred, y_test)                              # 14195
np.sqrt(mean_squared_error(y_pred,y_test))                       # 20558
r2_score(y_pred,y_test)                                          # 0.92

## Hence lookin at the rmse. r_sq values for training and test predictions of our model,
# we can say that our model is not overfitting and good to use. 
df["SalePrice"].describe()
# But still we have high rmse value





#################### SUPPORT VECTOR REGRESSION (SVR)  ###################




df= pd.read_csv("C:\\Users\\Soura\\Downloads\\DATA\\AMES_Final_DF.csv")
df.info()
df.columns

X= df.drop('SalePrice',axis=1)
y= df['SalePrice']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

## Scaling our data 
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

scaled_X_train= scaler.fit_transform(X_train)
scaled_X_test= scaler.transform(X_test)

from sklearn.svm import SVR

base_model= SVR()
base_model.fit(scaled_X_train, y_train)
base_pred= base_model.predict(scaled_X_test)
base_pred
base_pred.mean()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error(y_test,base_pred)
np.sqrt(mean_squared_error( y_test, base_pred))
r2_score(y_test, base_pred)
y_test.mean()

## Our model is predicting very badly with very high rmse and r2_score


#### now using grid search:::

param_grid= {"C":[0.001,0.01,0.1,0.5,1],
             'kernel':['linear','rbf','poly'],
             'gamma':['scale','auto'],'degree':[2,3,4], 
             'epsilon':[0,0.01,0.1,0.5,1,2]}

from sklearn.model_selection import GridSearchCV

svr= SVR()
grid= GridSearchCV(svr, param_grid)
grid.fit(scaled_X_train, y_train)

grid.best_params_
## grid.best_params_: {'C': 1, 'degree': 2, 'epsilon': 0, 'gamma': 'scale', 'kernel': 'linear'}

grid_pred= grid.predict(scaled_X_test)

mean_absolute_error(y_test, grid_pred)
np.sqrt(mean_squared_error(y_test, grid_pred))
r2_score(y_test, grid_pred)

## We have got a r2_score of 0.335 that is mediocre, rmse is also high so we keep exploring other models
 


###################### RANDOM FOREST MODEL ####################



df= pd.read_csv("C:\\Users\\Soura\\Downloads\\DATA\\AMES_Final_DF.csv")

X= df.drop('SalePrice',axis=1)
y= df['SalePrice']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

n_estimators= [64,100,128,200]
max_features= [2,3,4]
bootstrap= [True, False]
oob_score= [True, False]         ## It is not a substitute for accuracy, but only used in 
#case we are using bootstrap samples, so that we can utilize the unused rows for oob score 

param_grid={'n_estimators':n_estimators, 'max_features':max_features, 'bootstrap':bootstrap, 'oob_score':oob_score} 

# Base model
rfc= RandomForestRegressor()
grid= GridSearchCV(rfc, param_grid)

grid.fit(X_train, y_train)
grid.best_params_  ## grid.best_params_ {'bootstrap': False,'max_features': 4,'n_estimators': 200,'oob_score': False
 
### Building our model
rfc= RandomForestRegressor(n_estimators=200, max_features=2,bootstrap=True, oob_score=True)
rfc.fit(X_train, y_train)

rfc.oob_score_
predictions= rfc.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error( y_test, predictions))
r2_score(y_test, predictions)

## We got a good r2_score i.e. 0.85 hence this model is workable


































