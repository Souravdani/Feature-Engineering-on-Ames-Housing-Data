# Feature-Engineering-on-Ames-Housing-Data
The raw data of Ames housing features and sales has been modified to feed to a model using feature engineering.
The domain knowledge is very important in the feature engineering. We have txt file to have some knowledge about our dataframe.

> Dealing with Outliers
The outliers be due to variability in the measurement or it may occur due to an experimental error. An outlier can cause serious problem in statistical 
analysis. Hence, first we see the correlation of different features with our label i.e SalePrice
We have done a scatter plot on most correlated feartures and label to point out some extreme outliers and removed them beecause they are less in number 
and do not show any trend with respect to label.

> Dealing with missing data
We are checking the percentage of missing data for various features and plot them to visualize them to make decision and take further steps.
For features having missing values less than 1 percent, it is feasible here to drop because we have good number od data rows. 
Based on the Description Text File, Mas Vnr Type and Mas Vnr Area being missing (NaN) is likely to mean that house simply just doesn't have a masonry veneer, 
in which case, we will fill in this data as we did before.
Now focusing on other columns having na beyond one percent threshold set by us:
Filling the na values if na values are in substantial proportion of our dataset is a tricky work and statistical knowledge comes handy.

> Dealing with Categorical Data
We have to convert the features having categotical data into "dummy" variables, or known as "one-hot" encoding. 
We splitted our dataframe into two parts Numeric features and string features and create dummy variables for string features then concatenate back to 
give our final dataframe, which is ready for modelling.










