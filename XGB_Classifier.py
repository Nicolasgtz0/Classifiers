#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:20:24 2023

@author: nicolasgutierrez
"""

#Import packages 

import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn as skl


def read(filename:str):
    """
    Parameters
    ----------
    filename : str
        File path of credit risk dataset.

    Returns
    -------
    df : pd.dataframe
        This is the credit risk dataframe.

    """
    df = pd.read_csv(filename)
    
    df = df.apply(lambda x: x.astype(str).str.lower())
    df = df.replace('y', 1)
    df = df.replace('n', 0)
    
    df.drop(["person_home_ownership", "loan_intent","loan_grade"], axis = 1, inplace=True)
    
    df = df.astype('float32')
    df = df.dropna()
    return df 


def split(df):
    """

    Parameters
    ----------
    df : pd.dataframe
        Credit_risk dataframe.

    Returns
    -------
    y_train, y_test, X_train, X_test
    4 arrays containing train and test split of data 

    """
    xVar = np.asarray(df['loan_status'])
    yVar = np.asarray(df[['person_age','loan_int_rate','cb_person_default_on_file','loan_amnt']])
    
    y_train, y_test, X_train, X_test = skl.model_selection.train_test_split(yVar, xVar, test_size=0.2) 
    return y_train, y_test, X_train, X_test 


def fit_and_train(y_train:np.array, X_train:np.array):
    '''

    Parameters
    ----------
    y_train : np.array
        training features.
    X_train : np.array
        label features.

    Returns
    -------
    xgb : XGBoost()
        fitted model.
    '''
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(y_train, X_train)
    return xgb_model 



def predict(xgb_model:xgb.XGBClassifier, y_test:np.array):
    '''
    Parameters
    ----------
    xgb : XGBoost
        fitted model.
    y_test : np.array
        test features.

    Returns
    -------
    x_pred : np.array
        Array of the predicted labels.
    '''
    x_pred = xgb_model.predict(y_test)
    return x_pred


def asses(x_pred:np.array, X_test: np.array):  
    '''
    Parameters
    ----------
    x_pred : np.array
        Array of the predicted labels..
    X_test : np.array
        Labels test.

    Returns
    -------
    accuracy : float
        Matching accuracy, compare test to predicted.

    '''
    matching = np.where(x_pred == X_test)[0]
    accuracy = len(matching)/len(X_test)
    print('accuracy:', accuracy*100,'%')
    return accuracy

if __name__ =='__main__': 
    credit_risk = read('/Users/nicolasgutierrez/Desktop/Databases/data/credit_risk_dataset.csv')
    print(credit_risk)
    features_train, features_test, Label_train, Label_test = split(credit_risk)
    credit_model = fit_and_train(features_train,Label_train)
    predict_credit = predict(credit_model, features_test) 
    credit_accuracy = asses(predict_credit, Label_test)
