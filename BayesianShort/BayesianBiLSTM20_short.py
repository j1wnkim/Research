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


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size: 2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data =                     torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


# In[22]:


class BayesianModel(torch.nn.Module):
    def __init__(self, input_size, params1, params2, params3, num_layers, output_size):    
    #params1 = [conv_kernel_size, stride, max_kernel_size, LSTM_hidden_size, LSTM_num_layers]
    #params2 = [dropout_LSTM, dropout_FFN]
    #params3 = [hidden_size1, hidden_size2, hidden_size3]
        super(BayesianModel, self).__init__()        
        self.conv1D = torch.nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size = params1[0], stride = params1[1])
        self.max1D = torch.nn.MaxPool1d(kernel_size = params1[2], stride = params1[1])
        self.dropout1 = torch.nn.Dropout(params2[1])
        self.input_size = input_size - params1[0] - params1[2] + 2
        self.BiLSTM = LSTM(input_size = self.input_size, hidden_size = params1[3], dropout = params2[0], bidirectional = True, num_layers = params1[4])
        
        layers = []
        input_size = params1[3] *2 
        num_units = 0 
        for i in range(num_layers):
            num_units = params3[i] 
            layers.append(torch.nn.Linear(input_size, num_units, bias = True))
            #if i!= num_layers-1:
                #layers.append(torch.nn.BatchNorm1d(num_units)) # add in the batch norm. 
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(params2[1])) #also add in the dropout. 
            input_size = num_units
        
        self.intermediate_layers = torch.nn.Sequential(*layers)
        self.finalNN = torch.nn.Linear(num_units, output_size, bias = True)
        
    def forward(self, x):
        x = self.conv1D(x)
        x = self.max1D(x) 
        x = self.dropout1(x) 
        x = self.BiLSTM(x) 
        
        x = x[0]
        x = x[:, -1,:]
        
        x = self.intermediate_layers(x) 
        x = self.finalNN(x) 
        return x 


# In[30]:


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


def evaluate2(model, val_loader, criterion, criterion2, device):
    num_experiments = 100
    criterion2 = criterion2.to(device)
    with torch.no_grad():
        model.train()
        total_val_samples = 0;
        Validation_Loss_MAE = 0;
        Validation_Loss_MAPE = 0;
        predictions = []  
        std_list = [] 
        for val_input, val_output in val_loader:
            val_input = val_input.to(device);
            val_output = val_output.to(device);
            #Avgpred
            pred = model(val_input)
            pred = torch.squeeze(pred, 1)
            pred = torch.unsqueeze(pred, 0)
               
            for i in range(num_experiments - 1):
                predictedVal2 = model(val_input)
                predictedVal2 = torch.squeeze(predictedVal2, 1)
                predictedVal2 = torch.unsqueeze(predictedVal2, 0)
                pred = torch.cat([pred, predictedVal2], dim = 0)
                
            Avgpred = torch.mean(pred, dim = 0)
            stdev = torch.std(pred, dim = 0)
            predCsv = Avgpred.cpu().numpy()
            stdev = stdev.cpu().numpy()
            predictions.extend(predCsv) 
            std_list.extend(stdev)
            
            Validation_Loss_MAE += criterion(val_output, Avgpred) * val_output.size(0)
            Validation_Loss_MAPE += criterion2(val_output, Avgpred) * val_output.size(0)
            total_val_samples += val_output.size(0)
        Validation_Loss_MAE = Validation_Loss_MAE/total_val_samples
        Validation_Loss_MAPE = Validation_Loss_MAPE/total_val_samples 
        return Validation_Loss_MAE, Validation_Loss_MAPE, predictions, std_list   

def Train_and_Evaluate(train_loader, val_loader, device, params1, params2, params3, numEpochs, early_stop_epochs):
    model = BayesianModel(input_size = 12, params1 = params1, params2 = params2, params3 = params3, num_layers = 3, output_size = 1)
    model = model.to(device);
    LossFunction = torch.nn.L1Loss();
    best_val_loss = float('inf')
    Training_Loss = float('inf')
    early_stop_count = 0


    Optimizer = torch.optim.Adam(params = model.parameters())
    for epoch in range(0,numEpochs):
        model.train()
        Training_Loss = 0;
        total_samples = 0;
        for input, output in train_loader:
            input = input.to(device);
            #output = torch.squeeze(output, 1);
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
            total_val_samples = 0 
            Validation_Loss = 0 
            for val_input, val_output in val_loader:
                val_input = val_input.to(device)
                #val_output = torch.squeeze(val_output,1);
                val_output = val_output.to(device)
                predictedVal = model(val_input)
                predictedVal = torch.squeeze(predictedVal, 1)
                Validation_Loss += LossFunction(val_output, predictedVal) * val_output.size(0)
                total_val_samples += val_output.size(0)
            Validation_Loss = Validation_Loss/total_val_samples
            print("Validation Loss: ", Validation_Loss)

            if Validation_Loss < best_val_loss:
                best_val_loss = Validation_Loss
                torch.save(model, "/home/jik19004/FilesToRun/BayesianBiDirectional/BayesianBiLSTM20")
                early_stop_count = 0;   
            else:
                early_stop_count += 1
            if early_stop_count >= early_stop_epochs:
                return (Training_Loss, best_val_loss)
    return (Training_Loss, best_val_loss)

def predict(model, data_loader, device):
    #model.eval()
    predictions = []
    act_outputs = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            act_outputs.append(_.numpy())

    return (np.concatenate(predictions), np.concatenate(act_outputs))


def predict2(model, data_loader, device):
    #model.eval()
    predictions = []
    act_outputs = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = predict_with_dropout(model, data, device)
            predictions.append(output.cpu().numpy())
            act_outputs.append(_.numpy())

    return (np.concatenate(predictions), np.concatenate(act_outputs))


def predict_with_dropout(model, input_tensor, device):
    # Set the model to evaluation mode initially
    model.eval()
    # Manually enable dropout layers and ensure batchnorm layers are in eval mode
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Enable dropout
        elif isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
            module.eval()  # Ensure batchnorm is in eval mode
    
    # Perform the prediction
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)  # Add batch dimension if necessary
    
    return output


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
     
     
    if test_end < 91291:
        data = splitDataAndScale(WeatherData, DemandData, train_start, train_end, val_start, val_end, test_start, test_end)
        TrainingData = data[0]
        TrainingOutput = data[1]
        ValidationData = data[2]
        ValidationOutput = data[3]
        TestingData = data[4]
        TestingOutput = data[5]
        
        TrainingData = TimeSeriesDataset(np.array(TrainingData),np.array(TrainingOutput))
        ValidationData = TimeSeriesDataset(np.array(ValidationData), np.array(ValidationOutput))
        TestingData = TimeSeriesDataset(np.array(TestingData), np.array(TestingOutput))
        
        TrainingLoader = DataLoader(TrainingData, batch_size = 6, shuffle = True)
        ValidationLoader = DataLoader(ValidationData, batch_size = 3, shuffle = False) 
        TestingLoader = DataLoader(TestingData, batch_size = 3, shuffle = False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params1 = [3, 1, 3, 64, 1]
        params2 = [0.15, 0.15]
        params3 = [64, 32, 16]
        
        #params1 = [conv_kernel_size, stride, max_kernel_size, LSTM_hidden_size, LSTM_num_layers]
        #params2 = [dropout_LSTM, dropout_FFN]
        #params3 = [hidden_size1, hidden_size2, hidden_size3]

        numEpochs = 3000
        early_stop_epochs = 100
        train_str = train_output_start + "-" + train_output_end 
        val_str = val_output_start + "-" + val_output_end 
        test_str = test_output_start + "-" + test_output_end 
        
        print(train_str)
        print(val_str)
        print(test_str)
        
        best_val_loss = Train_and_Evaluate(TrainingLoader, ValidationLoader, device, params1, params2, params3, numEpochs, early_stop_epochs)
        TrainingLoader = DataLoader(TrainingData, batch_size = 6, shuffle = False)
        test_loss = evaluate2(torch.load("/home/jik19004/FilesToRun/BayesianBiDirectional/BayesianBiLSTM20"), TestingLoader, torch.nn.L1Loss(), MeanAbsolutePercentageError(), device) 
        val_loss = evaluate2(torch.load("/home/jik19004/FilesToRun/BayesianBiDirectional/BayesianBiLSTM20"), ValidationLoader, torch.nn.L1Loss(),MeanAbsolutePercentageError(), device)
        train_loss = evaluate2(torch.load("/home/jik19004/FilesToRun/BayesianBiDirectional/BayesianBiLSTM20"), TrainingLoader, torch.nn.L1Loss(),MeanAbsolutePercentageError(), device)
        
        
        print("Best train_loss: {}, Best val_loss: {}, Best test_loss: {}".format(train_loss, val_loss, test_loss))
        Training_Loss_MAE.append(train_loss[0].item())
        Validation_Loss_MAE.append(val_loss[0].item())
        Testing_Loss_MAE.append(test_loss[0].item())
        
        
        Training_Loss_MAPE.append(train_loss[1].item())
        Validation_Loss_MAPE.append(val_loss[1].item())
        Testing_Loss_MAPE.append(test_loss[1].item())
        
        df_train = pd.concat([df_train, pd.DataFrame({train_str: train_loss[2]})], ignore_index=False, axis=1)
        df_val = pd.concat([df_val, pd.DataFrame({val_str: val_loss[2]})], ignore_index=False, axis=1)
        df_test = pd.concat([df_test, pd.DataFrame({test_str: test_loss[2]})], ignore_index=False, axis=1)
        
        df_train_std = pd.concat([df_train_std, pd.DataFrame({train_str: train_loss[3]})], ignore_index=False, axis=1)
        df_val_std = pd.concat([df_val_std, pd.DataFrame({val_str: val_loss[3]})], ignore_index = False, axis =1 )
        df_test_std = pd.concat([df_test_std, pd.DataFrame({test_str: test_loss[3]})], ignore_index = False, axis = 1)
        

    else:
        break 

TrainingLoss_series = pd.DataFrame({"Train_MAE": Training_Loss_MAE, "Train_MAPE": Training_Loss_MAPE})
ValidationLoss_series = pd.DataFrame({"Validation_MAE": Validation_Loss_MAE, "Validation_MAPE": Validation_Loss_MAPE})
TestingLoss_series = pd.DataFrame({"Testing_MAE": Testing_Loss_MAE, "Testing_MAPE": Testing_Loss_MAPE})

TrainingLoss_series.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingLosses.csv", index = False)
ValidationLoss_series.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationLosses.csv", index = False)
TestingLoss_series.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingLosses.csv", index = False)

df_train.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingPredictions.csv", index = False)
df_val.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationPredictions.csv", index = False)
df_test.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingPredictions.csv", index = False)

df_train_std.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingStd.csv", index = False)
df_val_std.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationStd.csv", index = False)
df_test_std.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingStd.csv", index = False)

