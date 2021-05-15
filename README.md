# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains data about direct marketing campaigns (phone calls) from a Portugese banking institution. We will be using this dataset to determine the success of Bank Telemarketing, by seeking to predict whether the client will subscribe a term deposit (target variable y). The performance metric we were trying to optimize/maximize was accuracy, and based on this the best performing model was the AutoML Voting Ensemble model which yielded .917 accuracy in comparison to the top hyperdrive logistic regression model which yielded .909 accuracy. This is interesting as we only tune two hyperparameters for a relatively simple model (logistic regression) and the automl model only seemed to yield a marginal improvement. Additionally, accuracy is a pretty simplistic measure to assess model performance on, we should re-evaluate the use case to see if this makes the most sense.

## Scikit-learn Pipeline
The Data:
The source of the data is the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#. The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The dataset consists of 20 features and 32,950 rows with 3,692 positive and 29,258 negative classes.

Attribute Information:

Input variables:
bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

There's a handful of binary categorical variables, which will be numerically encoded. The months and weekdays are label encoded, but other than that there are no high cardinality categorical features in this dataset. There is class imbalance in the output variable though, so that will need to be rebalanced in future iterations.

The Pipeline:
The pipeline architecture involves an SKLearn estimator that gets the output of and is controlled by a train.py file. This train.py file reads in bankmarketing dataset, conducts feature engineering through OneHotEncoding the categorical variables and creating numerical features of the binary categorical variables in the dataset, and then splits the data into train/test splits of the features and target variable. The train.py file then takes the hyperparameter arguments passed in for C (the inverse of regularization strength) and max_iter (maximum iterations until convergence) and runs a Logistic Regression model. This model is scored for accuracy and then saved out to a file path. For hyperparameter tuning, we assess these values for C: .001, .01, .1, .3, .5, 1.0. and these for max_iter: 50, 100, 150, 200 through a random parameter sampler which selects a choice of one from each hyperparameter at random. We also have a bandit policy for termination criteria with a slack factor of .1 as the ratio threshold for the distance from the best performing experiment run. To achieve this hyperparameter tuning, the SKLearn estimator is passed into/wrapped by a HyperdriveConfig object. This is what allows the estimator to be run multiple times with different combinations of hyperparameters. The Logistic Regression classification model is a generalized linear model that uses a logistic function to model a binary dependent variable. This pipeline is set up to maximize accuracy.

The benefits of the random parameter sampler are primarily for computational cost reasons, and not as much for performance. It will randomly choose combinations of hyperparameters and supports early termination of low performance runs, which is good typically for an initial search. I would increase the runs and conduct the Bayesian sampling next as it will tune hyperparameters given the priors from previous runs. This is typically more costly and time intensive, though. 

The benefits of the early stopping policy I chose, the BanditPolicy, is that it specifically looks at the distance between the model performance evaluation metric from the best model based on some percentage and this is beneficial as we currently only have one performance metric listed. The BanditPolicy aims to terminate poorly performing runs by checking how far off the runs are from the best performing run. If we had multiple, the Median policy would likely be beneficial as it gets the median of the running averages and concurrently stops those with values lower than the median.

## AutoML
The AutoML models were 23 different models and they all were either a Gradient Boosted Decision Tree, Random Forest, or some other Tree Ensemble model. The best one was the Voting Ensemble. Ensemble modeling is a process where multiple diverse models are created with the aim to aggregate the predictions of each base model, resulting in one final prediction for the unseen data that is usually more accurate than the each one returned by each single model. Basically, ensembling strategies reduce the generalization error of the prediction. Voting Ensemble leverages soft-voting which uses weighted averages. The Voting Ensemble model beats out the next best by only .002 accuracy though, which is quite the marginal amount and can even be chalked up to stochasticity.

## Pipeline comparison
The AutoML VotingEnsemble model beats the Logistic Regression model by about .06 accuracy, which is also an incredibly marginal amount. I believe this kind of difference was likely due to model architecture as the VotingEnsemble model runs a lot of diverse models and then averages the results which has commonly yielded strong results. Logistic Regression model only runs for two hyperparameters and just selects the highest value; this quite simplistic architecture does well but is bound to get beaten by tree-based ensemble models.

## Future work
Even with the performance difference, this difference is small and that is likely because we don't have that many informative features & have class imbalance in the dataset. Feature engineering is bound to be able to extract the last bit of performance, and also handling the classification imbalance in the dataset. We don't have an equal, or relatively equal, number of values for our binary classification target values and this poses problem if we don't balance the dataset in the train.py file. Implementing SMOTE and creating stronger features is likely to improve performance quite a bit, and I would wager AutoML would outperform by a large margin after these improvements, especially because of the diversity and combination of the models leveraged.

The compute cluster delete code is in the ipynb file.
