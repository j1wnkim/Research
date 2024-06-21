#!/usr/bin/env python
# coding: utf-8

# # Energy Consumption Predictions with Bayesian LSTMs in PyTorch

# Author: Pawarit Laosunthara
# 
# 

# # **Important Note for GitHub Readers:**
# Please click the **Open in Colab** button above in order to view all **interactive visualizations**.
# 
# This notebook demonstrates an implementation of an (Approximate) Bayesian Recurrent Neural Network in PyTorch, originally inspired by the *Deep and Confident Prediction for Time Series at Uber* (https://arxiv.org/pdf/1709.01907.pdf)
# 
# <br>
# 
# In this approach, Monte Carlo dropout is used to **approximate** Bayesian inference, allowing our predictions to have explicit uncertainties and confidence intervals. This property makes Bayesian Neural Networks highly appealing to critical applications requiring uncertainty quantification.
# The *Appliances energy prediction* dataset used in this example is from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)
# 
# 
# **Note:** this notebook purely serves to demonstrate the implementation of Bayesian LSTMs (Long Short-Term Memory) networks in PyTorch. Therefore, extensive data exploration and feature engineering is not part of the scope of this investigation.

# # Preliminary Data Wrangling

# **Selected Columns:**
# 
# For simplicity and speed when running this notebook, only temporal and autoregressive features are used.
# 
# - date time year-month-day hour:minute:second, sampled every 10 minutes \
# - Appliances, energy use in Wh for the corresponding 10-minute timestamp \
# - day_of_week, where Monday corresponds to 0 \
# - hour_of_day
# 

# In[16]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 


# In[17]:


WeatherData = pd.read_csv('/home/jik19004/FilesToRun/finalized_weather_data.csv') 


# In[18]:


def return_sequences(data, outputData, input_n_steps, output_n_steps): 
    X = [] 
    Y = [] 
    length = len(data) 
    for i in range(0,length, 1): 
        input_indx = i + input_n_steps 
        output_indx = input_indx + output_n_steps  
        if (output_indx > len(data)): # we need to have equally split sequences.  
            break               # The remaining data that cannot fit into a fixed 
                                # sequence will immediately be cut! 
        else:
            Xsample = data.iloc[i:input_indx, :] # get the previous data 
            Ysample = outputData[input_indx:output_indx] 
            X.append(Xsample) 
            Y.append(Ysample) 
    X = np.asarray(X).astype('float64') 
    Y = np.asarray(Y).astype('float64') 
    return (X, Y) 


# In[19]:


def scaleTheData(data):
    scaler = StandardScaler()
    # split the data first. 
    data2 = scaler.fit_transform(data)
    data = pd.DataFrame(data2, columns = data.columns)
    return data

def splitDataAndScale(data, output, split = 61363):    
    TrainingData = scaleTheData(data.iloc[:split, :].copy()) # The index for Janruary 1st basically, don't necessarily hard code it. 
    # Janruary 1st 2018. 
    TrainingOutput = output[:split].copy()

    RemainingData = data.iloc[split:, :].copy()
    RemainingOutput = output[split:].copy()

    ValidationData = scaleTheData(RemainingData.iloc[:int(0.5*len(RemainingData)), :].copy())
    ValidationOutput = RemainingOutput[:int(0.5*len(RemainingOutput))].copy()

    TestingData = scaleTheData(RemainingData.iloc[int(0.5 * len(RemainingData)):, :].copy())
    TestingOutput = RemainingOutput[int(0.5 * len(RemainingData)):].copy()


    TrainingSequences = return_sequences(TrainingData, TrainingOutput, 18, 1) 

    TransformedTrainingData = TrainingSequences[0]
    TransformedTrainingOutput = TrainingSequences[1]

    ValidationSequences = return_sequences(ValidationData, ValidationOutput, 18, 1)

    TransformedValidationData = ValidationSequences[0]
    TransformedValidationOutput = ValidationSequences[1]

    TestingSequences = return_sequences(TestingData, TestingOutput, 18, 1)

    TransformedTestingData = TestingSequences[0]
    TransformedTestingOutput = TestingSequences[1]


    return (TransformedTrainingData, TransformedTrainingOutput, TransformedValidationData, TransformedValidationOutput, 
    TransformedTestingData, TransformedTestingOutput)


# ## Time Series Transformations
# 
# 1. The dataset is to be re-sampled at an hourly rate for more meaningful analytics.
# 
# 2. To alleviate exponential effects, the target variable is log-transformed as per the Uber paper.
# 
# 3. For simplicity and speed when running this notebook, only temporal and autoregressive features, namely `day_of_week`, `hour_of_day`, \
# and previous values of `Appliances` are used as features

# # Prepare Training Data

# For this example, we will use sliding windows of 10 points per each window (equivalent to 10 hours) to predict each next point. The window size can be altered via the `sequence_length` variable.
# 
# Min-Max scaling has also been fitted to the training data to aid the convergence of the neural network.

# In[20]:


LonData = (WeatherData.loc[:, "HVN_lon"] + WeatherData.loc[:, "IJD_lon"] + WeatherData.loc[:, "BDL_lon"] + WeatherData.loc[:, "HFD_lon"] 
+ WeatherData.loc[:, "BDR_lon"] + WeatherData.loc[:, "GON_lon"] + WeatherData.loc[:, "DXR_lon"] + WeatherData.loc[:, "MMK_lon"])/8
LatData = (WeatherData.loc[:, "HVN_lat"] + WeatherData.loc[:, "IJD_lat"] + WeatherData.loc[:, "BDL_lat"] + WeatherData.loc[:, "HFD_lat"] 
+ WeatherData.loc[:, "BDR_lat"] + WeatherData.loc[:, "GON_lat"] + WeatherData.loc[:, "DXR_lat"] + WeatherData.loc[:, "MMK_lat"])/8
TmpfData = (WeatherData.loc[:, "HVN_tmpf"] + WeatherData.loc[:, "IJD_tmpf"] + WeatherData.loc[:, "BDL_tmpf"] + WeatherData.loc[:, "HFD_tmpf"] 
+ WeatherData.loc[:, "BDR_tmpf"] + WeatherData.loc[:, "GON_tmpf"] + WeatherData.loc[:, "DXR_tmpf"] + WeatherData.loc[:, "MMK_tmpf"])/8
DrctData = (WeatherData.loc[:, "HVN_drct"] + WeatherData.loc[:, "IJD_drct"] + WeatherData.loc[:, "BDL_drct"] + WeatherData.loc[:, "HFD_drct"] 
+ WeatherData.loc[:, "BDR_drct"] + WeatherData.loc[:, "GON_drct"] + WeatherData.loc[:, "DXR_drct"] + WeatherData.loc[:, "MMK_drct"])/8
SkntData = (WeatherData.loc[:, "HVN_sknt"] + WeatherData.loc[:, "IJD_sknt"] + WeatherData.loc[:, "BDL_sknt"] + WeatherData.loc[:, "HFD_sknt"] 
+ WeatherData.loc[:, "BDR_sknt"] + WeatherData.loc[:, "GON_sknt"] + WeatherData.loc[:, "DXR_sknt"] + WeatherData.loc[:, "MMK_sknt"])/8
VsbyData = (WeatherData.loc[:, "BDR_vsby"] + WeatherData.loc[:, "IJD_vsby"] + WeatherData.loc[:, "BDL_vsby"] + WeatherData.loc[:, "HFD_vsby"] 
+ WeatherData.loc[:, "BDR_vsby"] + WeatherData.loc[:, "GON_vsby"] + WeatherData.loc[:, "DXR_vsby"] + WeatherData.loc[:, "MMK_vsby"])/8


WeatherData2 = pd.DataFrame({ "Hour": WeatherData.loc[:, "Hour"], "WeekDay or Weekend": WeatherData.loc[:, "WeekDay or Weekend"], "Sknt_hourly": SkntData,
                          "Tmpf_Hourly": TmpfData})


# In[21]:


DemandData = WeatherData['Demand'].copy() # The output data 
data = splitDataAndScale(WeatherData2, DemandData) # splitting the data into training, validation, and testing. 
TrainingData = data[0]
TrainingOutput = data[1]

ValidationData = data[2]
ValidationOutput = data[3]

TestingData = data[4]
TestingOutput = data[5]


# In[22]:


print(TrainingOutput.shape)
print(ValidationOutput.shape)
print(TestingOutput.shape)


# In[23]:


WeatherData2.head()


# # Define Bayesian LSTM Architecture

# To demonstrate a simple working example of the Bayesian LSTM, a model with a similar architecture and size to that in Uber's paper has been used a starting point. The network architecture is as follows:
# 
# Encoder-Decoder Stage:
#  - A uni-directional LSTM with 2 stacked layers & 128 hidden units acting as an encoding layer to construct a fixed-dimension embedding state
#  - A uni-directional LSTM with 2 stacked layers & 32 hidden units acting as a decoding layer to produce predictions at future steps
#  - Dropout is applied at **both** training and inference for both LSTM layers
# 
# 
#  Predictor Stage:
#  - 1 fully-connected output layer with 1 output (for predicting the target value) to produce a single value for the target variable
# 
# 
# By allowing dropout at both training and testing time, the model simulates random sampling, thus allowing varying predictions that can be used to estimate the underlying distribution of the target value, enabling explicit model uncertainties.
# 

# In[24]:


import torch 
class LSTMModel(torch.nn.Module):
    def __init__(self, num_layers, lastNeurons, act_slope, LSTMNeurons, params, output_num =1):
        super(LSTMModel, self).__init__()
        self.LSTM1 = torch.nn.LSTM(input_size = 4, hidden_size = LSTMNeurons, num_layers = 1, bias = True, batch_first = True)
        self.LSTM2 = torch.nn.LSTM(input_size = LSTMNeurons, hidden_size = LSTMNeurons, num_layers = 1, bias = True, batch_first = True, bidirectional = False)
        self.LSTM3 = torch.nn.LSTM(input_size = LSTMNeurons, hidden_size = LSTMNeurons, num_layers = 1, bias = True, batch_first = True, bidirectional = False)
        self.batchNorm0 = torch.nn.BatchNorm1d(num_features = 18)
        
        input_size = LSTMNeurons
        layers = [] 
        for i in range(num_layers):
            num_units = params[i]
            layers.append(torch.nn.Linear(input_size, num_units, bias = True))
            layers.append(torch.nn.BatchNorm1d(num_features = 18))
            layers.append(torch.nn.LeakyReLU(act_slope))
            layers.append(torch.nn.Dropout(0.4))
            input_size = num_units
        self.intermediateLayers = torch.nn.Sequential(*layers)
        self.Linear1 = torch.nn.Linear(in_features = input_size, out_features = lastNeurons, bias = True)
        self.Activation1 = torch.nn.LeakyReLU(act_slope)
        self.Dropout = torch.nn.Dropout(0.4)
        
        
        self.Linear2 = torch.nn.Linear(in_features = lastNeurons * 18, out_features = output_num, bias = True)
        self.Activation2 = torch.nn.LeakyReLU(act_slope)

        
        
        
    def forward(self, val):
        
        x = self.LSTM1(val)
        x = self.LSTM2(x[0], (x[1][0],x[1][1])) #(x[1][0], x[1][1])
        x = self.LSTM3(x[0], (x[1][0], x[1][1]))
        x = self.batchNorm0(x[0])
        x = self.Dropout(x)
        
        
        x = self.intermediateLayers(x)
        x = self.Linear1(x)
        x = self.Activation1(x)
        x = self.Dropout(x)
        x = x.view(-1, x.size(1) * x.size(2))  
              
        
        x = self.Linear2(x)
        x = self.Activation2(x)
        
        return x
        


# ### Begin Training

# To train the Bayesian LSTM, we use the ADAM optimizer along with mini-batch gradient descent (`batch_size = 128`). For quick demonstration purposes, the model is trained for 150 epochs.
# 
# The Bayesian LSTM is trained on the first 70% of data points, using the aforementioned sliding windows of size 10. The remaining 30% of the dataset is held out purely for testing.

# In[25]:


import optuna 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, output):
        data = torch.tensor(data).float(); 
        output = torch.tensor(output).float() 
        self.data = data 
        self.output = output; 

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        x = self.data[idx];  
        y = self.output[idx]; 
        
        return x, y; 

# use the past 72 hours in advance and then predict the 1st hour, 6th hour, 12 hours! 

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * target.size(0)
    return running_loss / len(val_loader.dataset)

def Train_and_Evaluate(train_loader, val_loader, device, params1, params2, numEpochs, early_stop_epochs):
    #num_layers, dropout = 0.1, outfeatures1 = 16, outfeatures2 = 16, outfeatures3 = 16, outfeatures4 = 16, dim_feedforward = 2048, output_num = 1
    model = LSTMModel(num_layers = params1[0], lastNeurons= params1[1], LSTMNeurons = params1[2], act_slope = params1[3], params = params2)
    model = model.to(device);
    LossFunction = torch.nn.L1Loss();
    best_val_loss = float('inf')
    early_stop_count = 0
   
    
    Optimizer = torch.optim.Adam(params = model.parameters(), weight_decay = 0.03)
    for epoch in range(0,numEpochs):
        model.train() 
        Training_Loss = 0; 
        total_samples = 0; 
        for input, output in train_loader:
            input = input.to(device); 
            output = torch.squeeze(output, 1); 
            output = output.to(device); 
            predictedVal = model(input)
            predictedVal = torch.squeeze(predictedVal, 1)
            Optimizer.zero_grad();
            batchLoss = LossFunction(predictedVal, output);
            batchLoss.backward(); 
            Optimizer.step();
            Training_Loss += batchLoss * output.size(0) #* output.size(0);
            total_samples += output.size(0)
        Training_Loss = Training_Loss.item()/total_samples


        Validation_Loss = 0; 
        print("passed ", epoch, "epoch", "Training Loss: ", Training_Loss," ", end = "")
        with torch.no_grad():
            model.eval()
            total_val_samples = 0; 
            Validation_Loss = 0; 
            for val_input, val_output in val_loader:
                val_input = val_input.to(device); 
                val_output = torch.squeeze(val_output,1);
                val_output = val_output.to(device);
                predictedVal = model(val_input)
                predictedVal = torch.squeeze(predictedVal, 1)
                Validation_Loss += LossFunction(val_output, predictedVal) * val_output.size(0)
                total_val_samples += val_output.size(0)
            Validation_Loss = Validation_Loss/total_val_samples
            print("Validation Loss: ", Validation_Loss)

            if Validation_Loss < best_val_loss:
                best_val_loss = Validation_Loss
                torch.save(model, "/home/jik19004/FilesToRun/BayesianLSTMs/LSTM1")
                early_stop_count = 0;
            else:
                early_stop_count +=1 
            if early_stop_count >= early_stop_epochs:
                return best_val_loss; 
    return best_val_loss; 

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    act_outputs = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            act_outputs.append(_.numpy())

    return (np.concatenate(predictions), np.concatenate(act_outputs))
    


# In[26]:


TrainingData = TimeSeriesDataset(np.array(TrainingData),np.array(TrainingOutput));
TrainingLoader = DataLoader(TrainingData, batch_size = 128);


ValidationData = TimeSeriesDataset(ValidationData, ValidationOutput); ### Set it with the previous validation data
ValidationLoader = DataLoader(ValidationData, batch_size = 128);


TestingData = TimeSeriesDataset(TestingData,TestingOutput); ### Set it with the previous testing data. 
TestingLoader = DataLoader(TestingData, batch_size = 256);


# In[28]:


def objective(trial):
    params1 = [trial.suggest_int("num_layers", low = 2, high = 4, step = 1),
              trial.suggest_int("last_hidden_neurons", low = 42, high = 90, step = 16),
              trial.suggest_int("LSTM_neurons", low = 72, high = 88, step = 8),
              trial.suggest_float("act_slope", low = 0.1, high = 0.3, log = True)]
    
    params2 = [trial.suggest_int("num_hiddenZero", low = 74, high = 138, step = 16),
               trial.suggest_int("num_hiddenOne", low = 106, high = 180, step = 16),
               trial.suggest_int("num_hiddenTwo", low = 58, high = 122, step = 16), 
               trial.suggest_int("num_hiddenThree", low = 58, high = 106, step = 16)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    return Train_and_Evaluate(TrainingLoader, ValidationLoader, device, params1, params2, 260, 35); 

import joblib 
study_name = 'sqlite:///LSTMOutput1.db'
study = optuna.create_study(direction = "minimize", sampler = optuna.samplers.TPESampler(), study_name = "NewLSTM1", load_if_exists = True, storage = 'sqlite:///LSTMOutput1.db')
joblib.dump(study, "LSTMOutput1.pkl")
study.optimize(objective, n_trials = 3500)


# In[ ]:


""

# In[ ]:


print(study.best_params)
print(study.best_value)


