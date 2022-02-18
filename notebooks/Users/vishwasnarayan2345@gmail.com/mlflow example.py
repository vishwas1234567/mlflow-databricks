# Databricks notebook source
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
from math import sqrt
import mlflow

# COMMAND ----------

!pip list

# COMMAND ----------

mlflow.set_experiment("/Users/vishwasnarayan2345@gmail.com/vishwasnew1")

# COMMAND ----------

pip install databricks-cli --upgrade

# COMMAND ----------

diabetes = datasets.load_diabetes()

# COMMAND ----------

diabetespd = pd.DataFrame(data=diabetes.data)
diabetespd.to_csv('diabetes.txt', encoding='utf-8', index=False)

# COMMAND ----------

diabetespd.head()

# COMMAND ----------

diabetes_X = diabetes.data[:, np.newaxis, 2]

# COMMAND ----------

with mlflow.start_run():
  # 1st idea
  diabetes_X = diabetes.data[:, np.newaxis, 2]
  
  # 2nd idea
  #diabetes_X = diabetes.data
  
  diabetes_X_train = diabetes_X[:-20]
  diabetes_X_test = diabetes_X[-20:]
 
  diabetes_y_train = diabetes.target[:-20]
  diabetes_y_test = diabetes.target[-20:]
 
  regr = linear_model
 
#   regr = linear_model.Lasso(alpha=0.1)
#   mlflow.log_param("alpha", 0.1)
  
  regr = linear_model.LassoLars(alpha=0.1)
  mlflow.log_param("alpha", 0.1)
 
  regr = linear_model.BayesianRidge()   
 
  regr.fit(diabetes_X_train, diabetes_y_train)
 
  diabetes_y_pred = regr.predict(diabetes_X_test)
 
  mlflow.log_metric("mse", mean_squared_error(diabetes_y_test, diabetes_y_pred))
  mlflow.log_metric("rmse", sqrt(mean_squared_error(diabetes_y_test, diabetes_y_pred)))
  mlflow.log_metric("r2", r2_score(diabetes_y_test, diabetes_y_pred))
  
  mlflow.log_artifact("diabetes.txt")

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# COMMAND ----------

with mlflow.start_run():
  
  import numpy as np
  from sklearn.linear_model import LinearRegression
  X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
  # y = 1 * x_0 + 2 * x_1 + 3
  y = np.dot(X, np.array([1, 2])) + 3
  
  reg = linear_model.LassoLars(alpha=0.1)
  mlflow.log_param("alpha", 0.1)
#   reg.score(X, y)
#   mlflow.log_param("alpha", 0.1)
  reg = LinearRegression().fit(X, y)
  reg.coef_
  reg.score(X, y)
 
  reg.intercept_
 
  reg.predict(np.array([[3, 5]]))

# COMMAND ----------

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from joblib import load
 
from logger import logger

# COMMAND ----------

import mlflow.sklearn
mlflow.sklearn.autolog()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
 
from numpy import savetxt
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# COMMAND ----------

db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

mlflow.sklearn.autolog()
 
with mlflow.start_run():
   
  n_estimators = 100
  max_depth = 6
  max_features = 3
  
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
 
  predictions = rf.predict(X_test)

# COMMAND ----------

with mlflow.start_run():
  
  n_estimators = 100
  max_depth = 6
  max_features = 3
  
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  predictions = rf.predict(X_test)
  
  mlflow.log_param("num_trees", n_estimators)
  mlflow.log_param("maxdepth", max_depth)
  mlflow.log_param("max_feat", max_features)
  
  mse = mean_squared_error(y_test, predictions)
    
  mlflow.log_metric("mse", mse)
  
  mlflow.sklearn.log_model(rf, "random-forest-model") 
  
  savetxt('predictions.csv', predictions, delimiter=',')
  
  mlflow.log_artifact("predictions.csv")
  
  df = pd.DataFrame(data = predictions - y_test)
  
 
  plt.plot(df)
  plt.xlabel("Observation")
  plt.ylabel("Residual")
  plt.title("Residuals")
 
  # Save the plot and log it as an artifact
  
  plt.savefig("residuals_plot.png")
  mlflow.log_artifact("residuals_plot.png") 

# COMMAND ----------

mlflow.sklearn.autolog()
with mlflow.start_run():
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy import stats
 
  from sklearn.linear_model import BayesianRidge, LinearRegression
 
  # #############################################################################
  # Generating simulated data with Gaussian weights
  np.random.seed(0)
  n_samples, n_features = 100, 100
  X = np.random.randn(n_samples, n_features)  # Create Gaussian data
  # Create weights with a precision lambda_ of 4.
  lambda_ = 4.
  w = np.zeros(n_features)
  # Only keep 10 weights of interest
  relevant_features = np.random.randint(0, n_features, 10)
  for i in relevant_features:
      w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
  # Create noise with a precision alpha of 50.
  alpha_ = 50.
#   mlflow.log_param(alpha_, alpha_)
  noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
  # Create the target
  
  
  y = np.dot(X, w) + noise
 
  # #############################################################################
  # Fit the Bayesian Ridge Regression and an OLS for comparison
  clf = BayesianRidge(compute_score=True)
  clf.fit(X, y)
 
  ols = LinearRegression()
  firval = ols.fit(X, y)
#   mlflow.log_param(firval, firval)
 
  # #############################################################################
  # Plot true weights, estimated weights, histogram of the weights, and
  # predictions with standard deviations
  lw = 2
  plt.figure(figsize=(6, 5))
  plt.title("Weights of the model")
  plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
           label="Bayesian Ridge estimate")
  plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
  plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
  plt.xlabel("Features")
  plt.ylabel("Values of the weights")
  plt.legend(loc="best", prop=dict(size=12))
 
  plt.figure(figsize=(6, 5))
  plt.title("Histogram of the weights")
  plt.hist(clf.coef_, bins=n_features, color='gold', log=True,
           edgecolor='black')
  plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
              color='navy', label="Relevant features")
  plt.ylabel("Features")
  plt.xlabel("Values of the weights")
  plt.legend(loc="upper left")
 
  plt.figure(figsize=(6, 5))
  plt.title("Marginal log-likelihood")
  plt.plot(clf.scores_, color='navy', linewidth=lw)
  plt.ylabel("Score")
  plt.xlabel("Iterations")
 
 
  # Plotting some predictions for polynomial regression
  def f(x, noise_amount):
      y = np.sqrt(x) * np.sin(x)
      noise = np.random.normal(0, 1, len(x))
      return y + noise_amount * noise
 
 
  degree = 10
  X = np.linspace(0, 10, 100)
  y = f(X, noise_amount=0.1)
  clf_poly = BayesianRidge()
  clf_poly.fit(np.vander(X, degree), y)
 
  X_plot = np.linspace(0, 11, 25)
  y_plot = f(X_plot, noise_amount=0)
  y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
  mlflow.log_param("y_mean", y_mean)
  mlflow.log_param("y_std", y_std)
  plt.figure(figsize=(6, 5))
  plt.errorbar(X_plot, y_mean, y_std, color='navy',
               label="Polynomial Bayesian Ridge Regression", linewidth=lw)
  plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
           label="Ground Truth")
  plt.ylabel("Output y")
  plt.xlabel("Feature X")
  plt.legend(loc="lower left")
  plt.show()

# COMMAND ----------

from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('Missing').getOrCreate()

# COMMAND ----------

training = spark.read.csv('/FileStore/tables/pimaindia.csv',header=True,inferSchema=True)

# COMMAND ----------

training.show()

# COMMAND ----------

training.printSchema()

# COMMAND ----------

training.columns

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=["BMI","SkinThickness"],outputCol="Independent Features")

# COMMAND ----------

output=featureassembler.transform(training)

# COMMAND ----------

output.show()

# COMMAND ----------

output.columns

# COMMAND ----------

finalized_data=output.select("Independent Features","DiabetesPedigreeFunction")

# COMMAND ----------

finalized_data.show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='DiabetesPedigreeFunction')
regressor=regressor.fit(train_data)

# COMMAND ----------

finalized_data.show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='DiabetesPedigreeFunction')
regressor=regressor.fit(train_data)

# COMMAND ----------

regressor.coefficients

# COMMAND ----------

regressor.intercept

# COMMAND ----------

pred_results=regressor.evaluate(test_data)

# COMMAND ----------

pred_results.predictions.show()

# COMMAND ----------

pred_results.meanAbsoluteError,pred_results.meanSquaredError

# COMMAND ----------

diabetes_df = spark.read.csv("/FileStore/tables/pimaindia.csv", header="true", inferSchema="true")
display(diabetes_df.groupBy("Glucose").avg("BMI").orderBy("Glucose"))

# COMMAND ----------

