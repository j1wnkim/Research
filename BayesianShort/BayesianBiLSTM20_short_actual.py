#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
from torchmetrics.regression import MeanAbsolutePercentageError


# In[2]:


import torch 

print(torch.__version__)


# In[3]:


data = pd.read_csv("/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv").drop(columns=["Unnamed: 0"])
data.head()


# In[4]:


from datetime import datetime 

WeatherData = pd.read_csv('/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv').drop(columns = ["Unnamed: 0"])
WeatherData.ffill(inplace = True)
WeatherData.bfill(inplace = True) # fill in missing values with the previous value.
DateTimeCol = WeatherData["Datetime"]
HourCol = []
WeekDayorWeekEndCol = [] 

for date in DateTimeCol:
    date = datetime.strptime(date, "%m/%d/%Y %H:%M")
    HourCol.append(date.hour)
    if date.weekday() < 5:
        WeekDayorWeekEndCol.append(0)
    else:
        WeekDayorWeekEndCol.append(1)

WeatherData.drop(columns = ["Datetime"], inplace = True) # drop the datetime column. 


WeatherData.insert(0, "Hour", HourCol)
WeatherData.insert(1, "Weekday or Weekend", WeekDayorWeekEndCol)


# In[5]:


DateTimeCol = [datetime.strptime(date, "%m/%d/%Y %H:%M") for date in DateTimeCol]
for i in range(len(DateTimeCol)):
    date = DateTimeCol[i]
    if int(date.year == 2018) and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 18: # start of short term data
        print("index for Dec 31, 2018: ", i)
    if int(date.year) == 2021 and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 23: # end of short term data  
        print("index for Dec 31, 2021: ", i)
    if int(date.year) == 2021 and int(date.month) == 1 and int(date.day) == 1 and int(date.hour) == 0: # start of validation
        print("index for Jan 1, 2021: ", i)
    if int(date.year) == 2022 and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 23: # end of validation 
        print("index for Dec 31, 2022 ", i)
        break



def return_sequences(data, outputData, input_n_steps, output_n_steps):
    X = []
    Y = []
    length = len(data)
    for i in range(0,length, 1):
        input_indx = i + input_n_steps
        #output_indx = input_indx + output_n_steps
        output_indx = i + output_n_steps
        if (input_indx > len(data)): # we need to have equally split sequences. >=
            break               # The remaining data that cannot fit into a fixed
                                # sequence will immediately be cut!
        else:
            Xsample = data.iloc[i:input_indx, :] # get the previous data
            #Ysample = outputData[input_indx:output_indx]
            Ysample = outputData[i]
            X.append(Xsample)
            Y.append(Ysample)
    X = np.asarray(X).astype('float64')
    Y = np.asarray(Y).astype('float64')
    return (X, Y)


# In[10]:


def splitDataAndScale(data, output, train_start = None, train_end = None, val_start = None, val_end = None, test_start = None, test_end = None):
    #train_end +=1
    #val_end += 1
    #test_end += 1
    TrainingData = (data.iloc[train_start: train_end + 1, :].copy())
    TrainingCategories = TrainingData.iloc[:, [0,1]]
    TrainingNumerical = TrainingData.iloc[:, 2:]
    TrainingOutput = output[train_start + 6: train_end + 2].copy()  
    Scaler = StandardScaler().fit(TrainingNumerical)
    TrainingNumerical = Scaler.transform(TrainingNumerical)
    TrainingCategories = TrainingCategories.reset_index(drop = True)
    TrainingData = pd.concat([TrainingCategories, pd.DataFrame(TrainingNumerical)], axis = 1)
    TrainingData.reset_index(drop = True, inplace = True)
    TrainingOutput.reset_index(drop = True, inplace = True)
    
    ValidationData = data.iloc[val_start: val_end + 1, :].copy()
    ValidationData.reset_index(drop = True, inplace = True)
    ValidationCategories = ValidationData.iloc[:, [0,1]]
    ValidationNumerical = ValidationData.iloc[:, 2:]
    ValidationNumerical = Scaler.transform(ValidationNumerical)
    ValidationCategories = ValidationCategories.reset_index(drop = True)
    ValidationData = pd.concat([ValidationCategories, pd.DataFrame(ValidationNumerical)], axis = 1)
    ValidationOutput = output[val_start + 6: val_end + 2].copy()
    ValidationData.reset_index(drop = True, inplace = True)
    ValidationOutput.reset_index(drop = True, inplace = True)
    
    TestingData = data.iloc[test_start: test_end + 1, :].copy()
    TestingData.reset_index(drop = True, inplace = True)
    TestingCategories = TestingData.iloc[:, [0,1]]
    TestingNumerical = TestingData.iloc[:, 2:]
    TestingNumerical = Scaler.transform(TestingNumerical)
    TestingCategories = TestingCategories.reset_index(drop = True)
    TestingData = pd.concat([TestingCategories, pd.DataFrame(TestingNumerical)], axis = 1)
    TestingOutput = output[test_start + 6: test_end + 2].copy()
    TestingData.reset_index(drop = True, inplace = True)
    TestingOutput.reset_index(drop = True, inplace = True)


    TrainingSequences = return_sequences(TrainingData, TrainingOutput, 6, 1)

    TransformedTrainingData = TrainingSequences[0]
    TransformedTrainingOutput = TrainingSequences[1]

    ValidationSequences = return_sequences(ValidationData, ValidationOutput, 6, 1)

    TransformedValidationData = ValidationSequences[0]
    TransformedValidationOutput = ValidationSequences[1]

    TestingSequences = return_sequences(TestingData, TestingOutput, 6, 1)

    TransformedTestingData = TestingSequences[0]
    TransformedTestingOutput = TestingSequences[1]


    return (TransformedTrainingData, TransformedTrainingOutput, TransformedValidationData, TransformedValidationOutput,
    TransformedTestingData, TransformedTestingOutput)


# In[11]:


from sklearn.preprocessing import StandardScaler 

DemandData = WeatherData['Demand'].copy() # The output data
WeatherData.drop(columns = ['Demand'], inplace = True)

# In[12]:


from torch.nn.utils.rnn import PackedSequence
from typing import *
import torch.nn as nn 


# In[22]:



# In[31]:


Training_Loss_MAE = [] 
Validation_Loss_MAE = [] 
Testing_Loss_MAE = [] 

Training_Loss_MAPE = [] 
Validation_Loss_MAPE = [] 
Testing_Loss_MAPE = [] 
# 87668
df_train = pd.DataFrame()
df_val = pd.DataFrame() 
df_test = pd.DataFrame()

df_train_std = pd.DataFrame()
df_val_std = pd.DataFrame()
df_test_std = pd.DataFrame()

DateTimeCol = pd.read_csv("/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv")["Datetime"]
ActualOutput = pd.read_csv("/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv")["Demand"] 
#70338
#91291
for i in range(70117, 91291, 24):
    train_start = i
    train_end = i + 25 * 6 - 2
    train_output_start = DateTimeCol[train_start + 6]
    train_output_end = DateTimeCol[train_end + 1]
    actual_train = ActualOutput[train_start + 6:train_end + 2] # 
    
    
    val_start = train_end + 1 #train_end - 4
    val_end = val_start + 28 #val_start + 28
    val_output_start = DateTimeCol[val_start + 6]
    val_output_end = DateTimeCol[val_end + 1]
    actual_val = ActualOutput[val_start + 6:val_end + 2] # 
    
    
    test_start = val_end + 1 #val_end - 4
    test_end = test_start + 28 #test_start + 28
    test_output_start = DateTimeCol[test_start + 6]
    test_output_end = DateTimeCol[test_end + 1]
    actual_test = ActualOutput[test_start + 6:test_end + 2] #
    actual_val = actual_val.to_numpy() 
    
    
    test_start = val_end + 1
    test_end = test_start + 28
    test_output_start = DateTimeCol[test_start + 6]
    test_output_end = DateTimeCol[test_end + 1]
    actual_test = ActualOutput[test_start + 6:test_end + 2] #
    actual_test = actual_test.to_numpy() 
    print("Computed at index {}".format(i))
     
    if test_end < 91291:
        train_str = train_output_start + "-" + train_output_end 
        val_str = val_output_start + "-" + val_output_end 
        test_str = test_output_start + "-" + test_output_end 
        
        df_train = pd.concat([df_train, pd.DataFrame({train_str: actual_train})], ignore_index=False, axis=1)
        df_val = pd.concat([df_val, pd.DataFrame({val_str: actual_val})], ignore_index=False, axis=1)
        df_test = pd.concat([df_test, pd.DataFrame({test_str: actual_test})], ignore_index=False, axis=1)
    else:
        break 


df_train.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingActual.csv", index = False)
df_val.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationActual.csv", index = False)
df_test.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingActual.csv", index = False)


