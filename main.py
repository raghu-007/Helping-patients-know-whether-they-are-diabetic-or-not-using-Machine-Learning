#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction using Machine Learning
# 
# Diabetes, is a group of metabolic disorders in which there are high blood sugar levels over a prolonged period. Symptoms of high blood sugar include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and damage to the eyes.
# 
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# ### **Objective**
# We will try to build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?
# 
# ### **Details about the dataset:**
# 
# The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# 
# - **Pregnancies**: Number of times pregnant
# - **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# - **BloodPressure**: Diastolic blood pressure (mm Hg)
# - **SkinThickness**: Triceps skin fold thickness (mm)
# - **Insulin**: 2-Hour serum insulin (mu U/ml)
# - **BMI**: Body mass index (weight in kg/(height in m)^2)
# - **DiabetesPedigreeFunction**: Diabetes pedigree function
# - **Age**: Age (years)
# - **Outcome**: Class variable (0 or 1)
# 
# **Number of Observation Units: 768**
# 
# **Variable Number: 9**
# 
# ### **Machine Learning Workflow**
# #### Steps
# 1. Data gathering
# 2. Data preperation
# 3. Exploratory Data Analysis
# 4. Data Preprocessing
# 5. Data Transformation
# 6. Model Building
# 7. Model Evaluation

# ## **1) Data Preperation and Exploratory Data Analysis**

# In[207]:


#Installation of required libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.simplefilter(action = "ignore") 


# In[208]:


#Reading the dataset
df = pd.read_csv("diabetes.csv")


# In[209]:


# The first 5 observation units of the data set were accessed.
df.head()


# In[210]:


# The size of the data set was examined. It consists of 768 observation units and 9 variables.
df.shape


# In[211]:


#Feature information
df.info()


# In[212]:


# Descriptive statistics of the data set accessed.
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99])


# In[213]:


# The distribution of the Outcome variable was examined.
df["Outcome"].value_counts()*100/len(df)


# In[214]:


# The classes of the outcome variable were examined.
df.Outcome.value_counts()


# In[215]:


# The histagram of the Age variable was reached.
# If you look at the histogram of the age variable, we can see that the histogram is quite skewed towards left, meaning \n
# data is not distributed properly, so we need to do some preprocessing, such as stadardizing/normalising.
plt.figure(figsize = (12,10))
df["Age"].hist(edgecolor = "black");
plt.xlabel("Age");
plt.ylabel("Frequency");
plt.title("Age Histogram");


# In[216]:


print("Max Age: " + str(df["Age"].max()) + " Min Age: " + str(df["Age"].min()))


# In[217]:


# Histogram and density graphs of all variables were accessed.
# Similar to previous chart, some density plots over here also shows signs of data skewdness, so we need to perform standardization
# before training our model, so that there wont be any bias while predicting the data.
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(df.BMI, bins = 20, ax=ax[3,1]) 


# In[218]:


# Plotting a Par Plot to see the relationship between the variables.
sns.pairplot(df)


# In[219]:


def group_mean(i):
    ab = df.groupby("Outcome").agg({i:"mean"})
    bc = df.groupby("Outcome").agg({"Insulin": "max"})
    return ab, bc
cd = df.drop(columns=["Outcome"], axis = 1)
for i in cd.columns:
    print(group_mean(i))


# In[220]:


df.groupby("Outcome").agg({"Pregnancies":"mean"})


# In[221]:


df.groupby("Outcome").agg({"Age":"mean"})


# In[222]:


df.groupby("Outcome").agg({"Age":"max"})


# In[223]:


df.groupby("Outcome").agg({"Insulin": "mean"})


# In[224]:


df.groupby("Outcome").agg({"Insulin": "max"})


# In[225]:


df.groupby("Outcome").agg({"Glucose": "mean"})


# In[226]:


df.groupby("Outcome").agg({"Glucose": "max"})


# In[227]:


df.groupby("Outcome").agg({"BMI": "mean"})


# In[228]:


# The distribution of the outcome variable in the data was examined and visualized.
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Outcome'].value_counts().plot.pie(explode=[0,0.08],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Target variable')
ax[0].set_ylabel('')
sns.countplot('Outcome',data=df,ax=ax[1])
ax[1].set_title('Outcome')
plt.show()


# In[229]:


# Below Data Explains about the correlation diagram of the data set.(How well each variable is related to each other.)
# If the correlation value is >0, there is a positive correlation. While the value of one variable increases, the value of the other variable also increases.
# Correlation = 0 means no correlation.
# If the correlation is <0, there is a negative correlation. While one variable increases, the other variable decreases. 
# When the correlations are examined, there are 2 variables that act as a positive correlation to the Outcome dependent variable.
# These variables are Glucose. As these increase, Outcome variable increases.
df.corr()


# In[230]:


# Correlation matrix graph of the data set
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=30)
plt.show()


# In[231]:


df.corrwith(df.Outcome).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target")


# ## **2) Data Preprocessing**

# #### 2.1) Eliminating Missing Values
# 
# As, We all were aware that, some values weere shown as 0, instead they are null values if we look at them closely, so having null values in or data might lead to a serious problem swhile training our ML Model, since it might give wrong predictions. So Below we replace 0 value by NaN:

# In[232]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[233]:


df.head()


# In[234]:


# Now, we can look at where are missing values
df.isnull().sum()


# In[235]:


# Have been visualized using the missingno library for the visualization of missing observations.
# Plotting A Bar Plot, so as to visualize count of null values, in each variable.
import missingno as msno
msno.bar(df);


# In[236]:


# OK, Now there are alot of techniques on how to fill null values, such as, using Simple Impouter, mean(), median, mode, etc.
# But we are going to analyse each variable, calculate the mean of each variable, with respect to the target variable and then we imputer values, based on the mean for each specific taregt calss.
# The missing values ​​will be filled with the median values ​​of each variable.
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# In[237]:


df['Insulin'].groupby(df['Outcome']).median()


# In[238]:


columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    print(median_target(i))


# In[239]:


df[df['Insulin'].notnull()]


# In[240]:


# The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick.
columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]


# In[241]:


df.head()


# In[242]:


# Missing values were filled.
df.isnull().sum()


# #### 2.2) Outlier Analysis: 
# Outliers are termed as extremely high or low values which are considered to be causwed due to noise/some kind of error.

# In[243]:


for i in df:
    print(i)


# In[244]:


df[df['Pregnancies'] > 10].any(axis = None)


# In[245]:


# In the data set, there were asked whether there were any outlier observations compared to the 25% and 75% quarters.
# It was found to be an outlier observation.
for feature in df:
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")


# In[246]:


# Visualizing the Insulin Variable in order to find out outliers, using boxplot.
plt.figure(figsize = (12,6))
sns.boxplot(x = df["Insulin"]);
plt.xlabel('Insulin')
plt.ylabel(' ')
plt.title('Outliers using Boxplot in Insulin Variable')


# In[247]:


# So inorde to remove these outliers, we supress them by replacing the outlier of that particular column with the Upper bound value, 
# similarly it can done with lower bound value.
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Insulin"] > upper,"Insulin"] = upper


# In[248]:


plt.figure(figsize = (12,6))
sns.boxplot(x = df["Insulin"]);
plt.xlabel('Insulin')
plt.ylabel(' ')
plt.title('Boxplot After removing outliers.')


# In[249]:


df.shape


# #### 2.3)  Local Outlier Factor (LOF)

# In[250]:


# We determine outliers between all variables with the LOF method
# The anomaly score of each sample is called the Local Outlier Factor. It measures the local deviation of the density of a given sample with respect to its neighbors. 
# It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood. 
from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors= 10)
lof.fit_predict(df)


# In[251]:



df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:30]


# In[252]:


#We choose the threshold value according to lof scores
threshold = np.sort(df_scores)[7]
threshold


# In[253]:


#We delete those that are higher than the threshold
outlier = df_scores > threshold
df = df[outlier]


# In[254]:


# The size of the data set was examined.
df.shape


# ## **3) Feature Engineering/ Feature Transformation.**
# 
# Creating new variables is important for models. But you need to create a logical new variable. For this data set, some new variables were created according to BMI, Insulin and glucose variables.

# In[255]:


# As we know BMI can be categorized into diffferent categories based on their values, such as given below.
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]


# In[256]:


df.head()


# In[257]:


# A categorical variable creation process is performed according to the insulin value.
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


# In[258]:


df['Insulin_Level'] = df.apply(set_insulin, axis = 1)
df.head()


# In[259]:


# Some intervals were determined according to the glucose variable and these were assigned categorical variables.
NewGlucose = pd.Series(["Low", "Normal", "Overweight", "High"], dtype = "category")
df["NewGlucose"] = NewGlucose
df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]


# In[260]:


df.head()


# #### 3.1) One Hot Encoding
# 
# Categorical variables in the data set should be converted into numerical values, Inorder to make the model understand the data. Inorde to do these we are using pd.get_dummies() to generate dummy variable for each category present in the data variable.

# In[261]:


# Here, by making One Hot Encoding transformation, categorical variables were converted into numerical values. It is also protected from the Dummy variable trap.
df = pd.get_dummies(df, columns =["NewBMI","Insulin_Level", "NewGlucose"], drop_first = True)


# In[262]:


df.head()


# In[263]:


categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'Insulin_Level_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight']]


# In[264]:


y = df["Outcome"]
X = df.drop(["Outcome", 'NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'Insulin_Level_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight'], axis = 1)
cols = X.columns
index = X.index


# In[265]:


X.head()


# In[266]:


# The variables in the data set are an effective factor in increasing the performance of the models by standardization.  
# There are multiple standardization methods. These are methods such as" Normalize"," MinMax"," Robust" and "Scale".
transformer = StandardScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns = cols, index = index)


# In[267]:


X.head()


# In[268]:


# Combine the Categorical variables unto the scaled data.
X = pd.concat([X,categorical_df], axis = 1)


# In[269]:


X.head()


# In[270]:


y.head()


# In[271]:


df = pd.concat([X,y], axis = 1)
df.head()


# ## **4) Model Building and Model Evaluation**

# In[272]:


X = df.drop(columns = ['Outcome'], axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[273]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# #### **1. Logistic Regression**

# In[274]:


lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


# #### **2. Decision Tree Classifer**

# In[275]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

print_score(dt_clf, X_train, y_train, X_test, y_test, train=True)
print_score(dt_clf, X_train, y_train, X_test, y_test, train=False)


# #### **3. Random Forest Classifier**

# In[276]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# #### **4. XGB(GRadient Boosting Classifier)**

# In[277]:


xgb_clf = GradientBoostingClassifier(random_state = 12345)
xgb_clf.fit(X_train, y_train)

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)


# ### **Testing on a Single User.**

# In[278]:


new = X_test.iloc[0]
a = np.asarray(new)
a = a.reshape(1,-1)
p = xgb_clf.predict(a)


# In[279]:


if (p[0] == 1):
    print("Person has Diabetes and is at risk of dying")
else:
    print("Great! the result is negative and you don't have to worry")


# # **8) Conclusion**
# 
# The aim of this study was to create classification models for the diabetes data set and to predict whether a person is affected by Diabetes or not, with the help of mahcine l;eanring and to obtain the best accuracy for our model which will help us to predict better results. The work flow is as follows:
# 
# 1) Reading an Understanding the Diabetes Dataset.
# 
# 2) With Exploratory Data Analysis; The data set's structural data were checked.
# The types of variables in the dataset were examined. Size information of the dataset was accessed. The 0 values in the data set are missing values. Primarily these 0 values were replaced with NaN values. Descriptive statistics of the data set were examined.
# 
# 3) Data Preprocessing section;
# df for: The NaN values missing observations were filled with the median values of whether each variable was sick or not. The outliers were determined by LOF and dropped. The X variables were standardized with the rubost method..
# 
# 4) During Model Building;
# At most, 4 algorithms were used and i would say, 4 of them have performed pretty well on out dataset, with better accuracy on both training and testing data.
# Algorithms: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier.
# 
# 5) Result;
# Out of 4, XGB has given us negligeble accuracy score, given as Training - 98.85% and Testing - 89.47%. 
# NoteL: Other algorithms also performed very well.

# 
