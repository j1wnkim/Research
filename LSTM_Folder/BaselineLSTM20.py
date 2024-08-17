#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn 


# In[8]:


data = pd.read_csv("/home/jik19004/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv").drop(columns=["Unnamed: 0"])
data.head()


# In[9]:


from datetime import datetime 

WeatherData = pd.read_csv('/home/jik19004/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv').drop(columns = ["Unnamed: 0"])
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


# In[10]:


DateTimeCol = [datetime.strptime(date, "%m/%d/%Y %H:%M") for date in DateTimeCol]
for i in range(len(DateTimeCol)):
    date = DateTimeCol[i]
    if int(date.year) == 2020 and int(date.month) == 1 and int(date.day) == 1 and int(date.hour) == 0: # end of training 
        print("index for Jan 1, 2020: ", i)
    if int(date.year) == 2021 and int(date.month) == 1 and int(date.day) == 1 and int(date.hour) == 0: # start of validation
        print("index for Jan 1, 2021: ", i)
    if int(date.year) == 2022 and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 23: # end of validation 
        print("index for Dec 31, 2022 ", i)
        break


# Add +1 to those indices

# In[11]:


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


# In[12]:


def splitDataAndScale(data, output, split_1 = 78883, split_2 = 87667, split_3 = 105187):
    TrainingData = (data.iloc[:split_1, :].copy())
    TrainingCategories = TrainingData.iloc[:, [0,1]]
    TrainingNumerical = TrainingData.iloc[:, 2:]
    TrainingOutput = output[:split_1].copy()  
    Scaler = StandardScaler().fit(TrainingNumerical)
    TrainingNumerical = Scaler.transform(TrainingNumerical)
    TrainingData = pd.concat([TrainingCategories, pd.DataFrame(TrainingNumerical)], axis = 1)
    
    ValidationData = data.iloc[split_2:split_3, :].copy()
    ValidationData.reset_index(drop = True, inplace = True)
    ValidationCategories = ValidationData.iloc[:, [0,1]]
    ValidationNumerical = ValidationData.iloc[:, 2:]
    ValidationNumerical = Scaler.transform(ValidationNumerical)
    ValidationData = pd.concat([ValidationCategories, pd.DataFrame(ValidationNumerical)], axis = 1)
    ValidationOutput = output[split_2:split_3].copy()
    
    TestingData = data.iloc[split_3:, :].copy()
    TestingData.reset_index(drop = True, inplace = True)
    TestingCategories = TestingData.iloc[:, [0,1]]
    TestingNumerical = TestingData.iloc[:, 2:]
    TestingNumerical = Scaler.transform(TestingNumerical)
    TestingData = pd.concat([TestingCategories, pd.DataFrame(TestingNumerical)], axis = 1)
    TestingOutput = output[split_3:].copy()


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

# In[13]:


DemandData = WeatherData['Demand'].copy() # The output data
WeatherData.drop(columns = ['Demand'], inplace = True)
data = splitDataAndScale(WeatherData, DemandData) # splitting the data into training, validation, and testing.


TrainingData = data[0]
TrainingOutput = data[1]

ValidationData = data[2]
ValidationOutput = data[3]

TestingData = data[4]
TestingOutput = data[5]


# In[14]:


print(TrainingOutput.shape)
print(ValidationOutput.shape)
print(TestingOutput.shape)


# In[15]:


print(TrainingData.shape)
print(ValidationData.shape)
print(TestingData.shape)


# In[16]:


print(TrainingData.shape)


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

# In[17]:


import torch
import torch.nn.functional as F
class BaselineLSTM(torch.nn.Module):
    def __init__(self, LSTMNeurons1, LSTMNeurons2, num_layers, params, output_num =1):
        super(BaselineLSTM, self).__init__()
        self.LSTM1 = torch.nn.LSTM(input_size = 12, hidden_size = LSTMNeurons1, num_layers = 1, bias = True, batch_first = True, bidirectional = False)
        self.LSTM2 = torch.nn.LSTM(input_size = LSTMNeurons1, hidden_size = LSTMNeurons2, num_layers = 1, bias = True, batch_first = True, bidirectional = False)

        #self.batchNorm0 = torch.nn.BatchNorm1d(num_features = LSTMNeurons)
        input_size = LSTMNeurons2
        layers = [] 
        for i in range(0,num_layers):
            num_units = params[i]
            layers.append(torch.nn.Linear(input_size, num_units, bias = True))
            if i!= num_layers -1:
                layers.append(torch.nn.BatchNorm1d(num_units))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            input_size = num_units
        self.intermediateLayers = torch.nn.Sequential(*layers)
        self.Linear1 = torch.nn.Linear(in_features = input_size, out_features = output_num, bias = True)

        
    def forward(self, val):
        
        x = self.LSTM1(val)
        x = self.LSTM2(x[0]) 

        x = x[0]
        x = x[:, -1, :]
        
        x = self.intermediateLayers(x)
        x = self.Linear1(x)
        
        return x


# ### Begin Training

# To train the Bayesian LSTM, we use the ADAM optimizer along with mini-batch gradient descent (`batch_size = 128`). For quick demonstration purposes, the model is trained for 150 epochs.
# 
# The Bayesian LSTM is trained on the first 70% of data points, using the aforementioned sliding windows of size 10. The remaining 30% of the dataset is held out purely for testing.

# In[18]:


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
    model = BaselineLSTM(LSTMNeurons1 = params1[0], LSTMNeurons2 = params1[1], num_layers = params1[2], params = params2)
    model = model.to(device);
    LossFunction = torch.nn.L1Loss();
    best_val_loss = float('inf')
    early_stop_count = 0


    Optimizer = torch.optim.Adam(params = model.parameters())
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
                torch.save(model, "/home/jik19004/FilesToRun/BaselineLSTM20/BaselineLSTM20")
                early_stop_count = 0;   
            else:
                early_stop_count +=1
            if early_stop_count >= early_stop_epochs:
                return best_val_loss;
    return best_val_loss;

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


# In[19]:


def Train_and_Evaluate2(train_loader, val_loader, device, params1, params2, numEpochs, early_stop_epochs):
    #num_layers, input_dim, hidden_unit1, hidden_unit2, output_unit, lastNeurons, batch_size, params, device = None
    model = BaselineLSTM(LSTMNeurons1 = params1[0], LSTMNeurons2 = params1[1], num_layers = params1[2], params = params2)
    model = model.to(device);
    TrainEpochLoss = [] 
    ValidationEpochLoss = [] 
    LossFunction = torch.nn.L1Loss();
    best_val_loss = float('inf')
    early_stop_count = 0


    Optimizer = torch.optim.Adam(params = model.parameters())
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
        TrainEpochLoss.append(Training_Loss)


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
            Validation_Loss = Validation_Loss.item()/total_val_samples
            print("Validation Loss: ", Validation_Loss)
            ValidationEpochLoss.append(Validation_Loss)

            if Validation_Loss < best_val_loss:
                best_val_loss = Validation_Loss
                torch.save(model, "/home/jik19004/FilesToRun/BaselineLSTM20/BaselineLSTM20")
                early_stop_count = 0;   
            else:
                early_stop_count +=1
            if early_stop_count >= early_stop_epochs:
                return (TrainEpochLoss, ValidationEpochLoss);
    return (TrainEpochLoss, ValidationEpochLoss);


# In[20]:


TrainingData = TimeSeriesDataset(np.array(TrainingData),np.array(TrainingOutput));
TrainingLoader = DataLoader(TrainingData, batch_size = 256);


ValidationData = TimeSeriesDataset(ValidationData, ValidationOutput); ### Set it with the previous validation data
ValidationLoader = DataLoader(ValidationData, batch_size = 256);


TestingData = TimeSeriesDataset(TestingData,TestingOutput); ### Set it with the previous testing data.
TestingLoader = DataLoader(TestingData, batch_size = 256);


# In[21]:


#num_layers, hidden_unit1, hidden_unit2, lastNeurons, batch_size, params, device = None
def objective(trial):
    params1 = [trial.suggest_int("LSTM_neurons1", low = 158, high = 256, step = 14),
               trial.suggest_int("LSTM_neurons2", low = 158, high = 256, step = 14),
              trial.suggest_int("num_layers", low = 1, high = 4, step = 1)]
    params2 = [trial.suggest_int("num_hiddenZero", low = 72, high = 180, step = 36),
               trial.suggest_int("num_hiddenOne", low = 72, high = 180, step = 36),
               trial.suggest_int("num_hiddenTwo", low = 58, high = 154, step = 24),
               trial.suggest_int("num_hiddenThree", low = 58, high = 154, step = 24)]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    return Train_and_Evaluate(TrainingLoader, ValidationLoader, device, params1, params2, 2000, 150); 


import joblib
study_name = 'sqlite:///LSTMBaselineOutput20.db'
study = optuna.create_study(direction = "minimize", sampler = optuna.samplers.TPESampler(), study_name = "NewLSTMBaseline20", load_if_exists = True, storage = 'sqlite:///LSTMBaselineOutput20.db')
study.optimize(objective, n_trials = 400)


# In[22]:


print(study.best_params)
print(study.best_value)


# In[23]:


no


# In[ ]:


params1 = [study.best_params["hidden_size1"], study.best_params["hidden_size2"], study.best_params["num_layers"], study.best_params["act_slope"]]
params2 = [study.best_params["num_hiddenZero"], study.best_params["num_hiddenOne"], study.best_params["num_hiddenTwo"], study.best_params["num_hiddenThree"]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
#best_val_loss = Train_and_Evaluate2(TrainingLoader, ValidationLoader, device, params1, params2, 2000, 150)


# In[29]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchmetrics import MeanAbsolutePercentageError

model = torch.load("/home/jik19004/FilesToRun/BayesianLSTM20/BayesianLSTM20BEST(NOW)")
predictions = predict(model, TestingLoader, device = torch.device("cuda"))
MAE_result = mean_absolute_error(predictions[0], predictions[1])
MSE_result = mean_squared_error(predictions[0], predictions[1])
MAPE = MeanAbsolutePercentageError() 
MAPE_result = MAPE(torch.Tensor(predictions[0]), torch.Tensor(predictions[1]))
print("MSE for Testing: ", MSE_result)
print("MAE for Testing: ", MAE_result)
print("MAPE for Testing: ", MAPE_result)


# In[30]:


predictions = predict(model, ValidationLoader, device = torch.device("cuda"))
MAE_result = mean_absolute_error(predictions[0], predictions[1])
MAPE = MeanAbsolutePercentageError()
MAPE_result = MAPE(torch.Tensor(predictions[0]), torch.Tensor(predictions[1]))

print("MAE for Validation: ", MAE_result)
print("MAPE for Validation: ", MAPE_result)


# In[31]:


predictions = predict(model, TrainingLoader, device = torch.device("cuda"))
MAE_result = mean_absolute_error(predictions[0], predictions[1])
MAPE = MeanAbsolutePercentageError()
MAPE_result = MAPE(torch.Tensor(predictions[0]), torch.Tensor(predictions[1]))


print("MAE for Training: ", MAE_result)
print("MAPE for Training: ", MAPE_result)


# In[ ]:


import matplotlib.pyplot as plt 

TrainEpochLoss = best_val_loss[0]
ValidationEpochLoss = best_val_loss[1]
fig = plt.figure()
axes = fig.add_axes([0,0,1.7,1.25])
axes.plot(TrainEpochLoss, label = "Training Loss")
axes.plot(ValidationEpochLoss, label = "Validation Loss")
axes.set_yticks(np.arange(0, 3000, 200))
axes.set_title("Training and Validation Loss for Bayesian LSTM")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss: Mean Absolute Error")
axes.legend()


# # Evaluating Model Performance

# The Bayesian LSTM implemented is shown to produce reasonably accurate and sensible results on both the training and test sets, often comparable to other existing frequentist machine learning and deep learning methods.
# 
# 

# In[32]:


x = pd.read_csv('/home/jik19004/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv').drop(columns = ["Unnamed: 0"])
x.ffill(inplace = True)
x.bfill(inplace = True) # fill in missing values with the previous value.
x


# In[33]:


import pandas as pd
from datetime import datetime 
x = pd.read_csv("/home/jik19004/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv")["Datetime"]
dateList = []
for i in range(len(x)): ## just create the datetimes so that we can use it for graph plotting.
    dateList.append(datetime.strptime(x[i], "%m/%d/%Y %H:%M"))


# In[34]:


x = np.array(pd.read_csv("/home/jik19004/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv")["Demand"])
x = pd.DataFrame({"DateTime": dateList, "Demand": x})


# In[35]:


training_df = x.iloc[:78883, :]
validation_df = x.iloc[87667:105187, :]
testing_df = x.iloc[105187:, :] #splits for training, validation, and testing.


# # Uncertainty Quantification

# The fact that stochastic dropouts are applied after each LSTM layer in the Bayesian LSTM enables users to interpret the model outputs as random samples from the posterior distribution of the target variable.
# 
# This implies that by running multiple experiments/predictions, can approximate  parameters of the posterioir distribution, namely the mean and the variance, in order to create confidence intervals for each prediction.
# 
# In this example, we construct 99% confidence intervals that are three standard deviations away from the approximate mean of each prediction.

# In[37]:


bayesian_lstm = torch.load("/home/jik19004/FilesToRun/BayesianLSTM20/BayesianLSTM20BEST(NOW)")
print(bayesian_lstm)


# In[38]:


TrainingPredictions, _ = predict(bayesian_lstm, TrainingLoader, torch.device("cuda"))
ValidationPredictions, _ = predict(bayesian_lstm, ValidationLoader, torch.device("cuda"))
TestingPredictions, _ = predict(bayesian_lstm, TestingLoader, torch.device("cuda"))


# In[39]:


x


# In[40]:


training_df = pd.DataFrame()
validation_df = pd.DataFrame()
testing_df = pd.DataFrame()

training_df["Date"] = x.iloc[18:78883, 0]
training_df["Actual Train Output"] = x.iloc[18:78883, 1]
training_df["Predicted Train Output"] = TrainingPredictions

validation_df["Date"] = x.iloc[87667+18:105187,0]
validation_df["Actual Val Output"] = x.iloc[87667+18:105187, 1]
validation_df["Predicted Val Output"] = ValidationPredictions

testing_df["Date"] = x.iloc[105187 + 18 :, 0]
testing_df["Actual Test Output"] = x.iloc[105187 + 18:, 1]
testing_df["Predicted Test Output"] = TestingPredictions


# In[41]:


n_experiments = 500


test_uncertainty_df = pd.DataFrame()
test_uncertainty_df['Date'] = testing_df['Date']


for i in range(n_experiments):
  experiment_predictions, _ = predict(bayesian_lstm, TestingLoader, torch.device("cuda"))
  test_uncertainty_df['energy_demand_{}'.format(i)] = experiment_predictions


energy_consumption_df = test_uncertainty_df.filter(like='energy_demand', axis=1)
test_uncertainty_df2 = test_uncertainty_df.copy() # copy and creat a duplicate of the dataframe. 

test_uncertainty_df['energy_demand_mean'] = energy_consumption_df.mean(axis=1)
test_uncertainty_df['energy_demand_std'] = energy_consumption_df.std(axis=1)

test_uncertainty_df = test_uncertainty_df[['Date', 'energy_demand_mean', 'energy_demand_std']]


# In[54]:


test_uncertainty_df2.to_csv("Testing500.csv", index = False)


# In[42]:


arr = np.array(test_uncertainty_df['energy_demand_mean'])
actualVals = np.array(testing_df["Actual Test Output"])

error = mean_absolute_error(arr, actualVals)
mape_error = MAPE(torch.Tensor(arr), torch.Tensor(actualVals))

print(error)
print(mape_error)


# In[43]:


n_experiments = 500


validation_uncertainty_df = pd.DataFrame()
validation_uncertainty_df['Date'] = validation_df['Date']


for i in range(n_experiments):
  experiment_predictions, _ = predict(bayesian_lstm, ValidationLoader, torch.device("cuda"))
  validation_uncertainty_df['energy_demand_{}'.format(i)] = experiment_predictions


energy_consumption_df = validation_uncertainty_df.filter(like='energy_demand', axis=1)
validation_uncertainty_df2 = validation_uncertainty_df.copy() # copy and creat a duplicate of the dataframe. 

validation_uncertainty_df['energy_demand_mean'] = energy_consumption_df.mean(axis=1)
validation_uncertainty_df['energy_demand_std'] = energy_consumption_df.std(axis=1)

validation_uncertainty_df = validation_uncertainty_df[['Date', 'energy_demand_mean', 'energy_demand_std']]


# In[55]:


validation_uncertainty_df2.to_csv("Validation500.csv", index = False)


# In[44]:


arr = np.array(validation_uncertainty_df['energy_demand_mean'])
actualVals = np.array(validation_df["Actual Val Output"])

error = mean_absolute_error(arr, actualVals)
mape_error = MAPE(torch.Tensor(arr), torch.Tensor(actualVals))

print(error)
print(mape_error)


# In[45]:


n_experiments = 500


training_uncertainty_df = pd.DataFrame()
training_uncertainty_df['Date'] = training_df['Date']


for i in range(n_experiments):
  experiment_predictions, _ = predict(bayesian_lstm, TrainingLoader, torch.device("cuda"))
  training_uncertainty_df['energy_demand_{}'.format(i)] = experiment_predictions


energy_consumption_df = training_uncertainty_df.filter(like='energy_demand', axis=1)
training_uncertainty_df2 = training_uncertainty_df.copy() # copy and creat a duplicate of the dataframe. 

training_uncertainty_df['energy_demand_mean'] = energy_consumption_df.mean(axis=1)
training_uncertainty_df['energy_demand_std'] = energy_consumption_df.std(axis=1)

training_uncertainty_df = training_uncertainty_df[['Date', 'energy_demand_mean', 'energy_demand_std']]


# In[56]:


training_uncertainty_df2.to_csv("Training500.csv", index = False)


# In[46]:


arr = np.array(training_uncertainty_df['energy_demand_mean'])
actualVals = np.array(training_df["Actual Train Output"])

error = mean_absolute_error(arr, actualVals)
mape_error = MAPE(torch.Tensor(arr), torch.Tensor(actualVals))

print(error)
print(mape_error)


# In[47]:


test_uncertainty_df['99% lower_bound'] = test_uncertainty_df['energy_demand_mean'] - 3*test_uncertainty_df['energy_demand_std'] #99 % confidence interval
test_uncertainty_df['99% upper_bound'] = test_uncertainty_df['energy_demand_mean'] + 3*test_uncertainty_df['energy_demand_std'] #99 % confidene interval

test_uncertainty_df["99.99% lower_bound"] = test_uncertainty_df["energy_demand_mean"] - 3.15 * test_uncertainty_df["energy_demand_std"]
test_uncertainty_df["99.99% upper_bound"] = test_uncertainty_df["energy_demand_mean"] + 3.15 * test_uncertainty_df["energy_demand_std"]

test_uncertainty_df["99.999% lower_bound"] = test_uncertainty_df["energy_demand_mean"] - 3.3 * test_uncertainty_df["energy_demand_std"]
test_uncertainty_df["99.999% upper_bound"] = test_uncertainty_df["energy_demand_mean"] + 3.3 * test_uncertainty_df["energy_demand_std"]


# In[ ]:


import plotly.graph_objects as go

test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['Date'].between('2022-01-04 06:00:00', '2022-1-06 23:00:00')]
truth_uncertainty_plot_df = testing_df.copy(deep=True)
truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_df['Date'].between('2022-01-04 06:00:00', '2022-1-06 23:00:00')]

upper_trace = go.Scatter(
    x=test_uncertainty_plot_df['Date'],
    y=test_uncertainty_plot_df['99% upper_bound'],
    mode='lines',
    fill=None,
    name='99 Upper Confidence Bound'
    )
lower_trace = go.Scatter(
    x=test_uncertainty_plot_df['Date'],
    y=test_uncertainty_plot_df['99% lower_bound'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 211, 0, 0.1)',
    name='99 lower Confidence Bound'
    )
real_trace = go.Scatter(
    x=truth_uncertainty_plot_df['Date'],
    y=truth_uncertainty_plot_df['Actual Test Output'],
    mode='lines',
    fill=None,
    name='Real Values'
    )

data = [upper_trace, lower_trace, real_trace]

fig = go.Figure(data=data)
fig.update_layout(title='Uncertainty Quantification for Energy Consumption Test Data',
                   xaxis_title='Time',
                   yaxis_title='Energy Demand (kWh)')

fig.show()


# #### Evaluating Uncertainty

# Using multiple experiments above, 99% confidence intervals have been constructed for each the prediction of the target variable (the logarithm of appliance power consumption). While we can visually observe that the model is generally capturing the behavior of the time-series, approximately only 50% of the real data points lie within a 99% confidence interval from the mean prediction value.
# 
# Despite the relatively low percentage of points within the confidence interval, it must be noted that Bayesian Neural Networks only seek to quantify the epistemic model uncertainty and does not account for aleatoric uncertainty (i.e. noise).

# In[58]:


bounds_df = pd.DataFrame()

# Using 99% confidence bounds

bounds_df['lower_bound'] = test_uncertainty_df['99.999% lower_bound']
bounds_df['prediction'] = test_uncertainty_df['energy_demand_mean']
bounds_df['real_value'] = testing_df["Actual Test Output"]
bounds_df['upper_bound'] = test_uncertainty_df['99.999% upper_bound']

bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                          (bounds_df['real_value'] <= bounds_df['upper_bound']))

print("Proportion of points for testing contained within 99.999% confidence interval:",
      bounds_df['contained'].mean())


# In[49]:


print(len(bounds_df))


# In[50]:


validation_uncertainty_df['99% lower_bound'] = validation_uncertainty_df['energy_demand_mean'] - 3*validation_uncertainty_df['energy_demand_std'] #99 % confidence interval
validation_uncertainty_df['99% upper_bound'] = validation_uncertainty_df['energy_demand_mean'] + 3*validation_uncertainty_df['energy_demand_std'] #99 % confidene interval

validation_uncertainty_df['99.99% lower_bound'] = validation_uncertainty_df['energy_demand_mean'] - 3.15*validation_uncertainty_df['energy_demand_std'] #99 % confidence interval
validation_uncertainty_df['99.99% upper_bound'] = validation_uncertainty_df['energy_demand_mean'] + 3.15*validation_uncertainty_df['energy_demand_std'] #99 % confidene interval

validation_uncertainty_df['99.999% lower_bound'] = validation_uncertainty_df['energy_demand_mean'] - 3*validation_uncertainty_df['energy_demand_std'] #99 % confidence interval
validation_uncertainty_df['99.999% upper_bound'] = validation_uncertainty_df['energy_demand_mean'] + 3*validation_uncertainty_df['energy_demand_std'] #99 % confidene interval


# In[51]:


bounds_df = pd.DataFrame()

# Using 99% confidence bounds
bounds_df['lower_bound'] = validation_uncertainty_df['99% lower_bound']
bounds_df['prediction'] = validation_uncertainty_df['energy_demand_mean']
bounds_df['real_value'] = validation_df['Actual Val Output']
bounds_df['upper_bound'] = validation_uncertainty_df['99% upper_bound']

bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                          (bounds_df['real_value'] <= bounds_df['upper_bound']))

print("Proportion of points for validation contained within 99% confidence interval:",
      bounds_df['contained'].mean())


# In[52]:


training_uncertainty_df['99% lower_bound'] = training_uncertainty_df['energy_demand_mean'] - 3*training_uncertainty_df['energy_demand_std'] #99 % confidence interval
training_uncertainty_df["99% upper_bound"] = training_uncertainty_df['energy_demand_mean'] + 3*training_uncertainty_df['energy_demand_std'] #99 % confidene interval

training_uncertainty_df['99.99% lower_bound'] = training_uncertainty_df['energy_demand_mean'] - 3.15*training_uncertainty_df['energy_demand_std'] #99 % confidence interval
training_uncertainty_df["99.99% upper_bound"] = training_uncertainty_df['energy_demand_mean'] + 3.15*training_uncertainty_df['energy_demand_std'] #99 % confidene interval

training_uncertainty_df['99.999% lower_bound'] = training_uncertainty_df['energy_demand_mean'] - 3.3*training_uncertainty_df['energy_demand_std'] #99 % confidence interval
training_uncertainty_df["99.999% upper_bound"] = training_uncertainty_df['energy_demand_mean'] + 3.3*training_uncertainty_df['energy_demand_std'] #99 % confidene interval


# In[53]:


bounds_df = pd.DataFrame()

bounds_df['lower_bound'] = training_uncertainty_df['99.999% lower_bound']
bounds_df['prediction'] = training_uncertainty_df['energy_demand_mean']
bounds_df['real_value'] = training_df['Actual Train Output']
bounds_df['upper_bound'] = training_uncertainty_df['99.999% upper_bound']

bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                            (bounds_df['real_value'] <= bounds_df['upper_bound']))

print("Proportion of points for training contained within 99.999% confidence interval:",
        bounds_df['contained'].mean())


# # Analyzing the Error 

# In[ ]:


training_uncertainty


# In[ ]:


training_df["Actual Train Output"].iloc[0]


# In[ ]:


print(type(training_uncertainty_df["energy_demand_mean"].iloc[0]))


# In[ ]:


training_uncertainty_df = training_uncertainty_df.copy()
training_uncertainty_df.reset_index(drop = True, inplace = True)
Errors = []

index = 0 
for prediction in training_uncertainty_df["energy_demand_mean"]:
    Errors.append(mean_absolute_error([prediction], [np.float32(training_df["Actual Train Output"].iloc[index])]))
    index+=1 

TrainingErrors = pd.DataFrame({"Date": training_uncertainty_df["Date"], "Errors": Errors})
print(f"The average training error: {np.mean(Errors)}")
print(f"The standard deviation of the training error: ", np.std(Errors))


# In[ ]:


MonthErrors = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}

for i in range(len(training_uncertainty_df["Date"])):
    date = training_uncertainty_df["Date"].iloc[i]
    MonthErrors[date.month].append(Errors[i])
    
for key in MonthErrors.keys():
    mean = np.mean(MonthErrors[key])
    std = np.std(MonthErrors[key])
    MonthErrors[key] = (mean, std)
    print(f"Month {key}: Mean Error: {mean}, Standard deviation: {std}")


# In[ ]:



# Assuming MonthErrors is a dictionary where:
# - keys are months
# - values are lists where the first element is the mean error and the second element is the standard deviation

# Extracting months, mean errors, and standard deviations
months = list(MonthErrors.keys())
mean_errors = [MonthErrors[key][0] for key in months]
std_devs = [MonthErrors[key][1] for key in months]
error_bars = [(0, std) for std in std_devs]

# Creating the bar plot with error bars
plt.bar(months, mean_errors, yerr=error_bars, capsize=5)  # capsize specifies the width of the horizontal line at the top of the error bar

plt.xlabel("Month")
plt.ylabel("Error in MWH")
plt.title("Monthly Error with Standard Deviation for Training")

plt.show()


# In[ ]:


print(validation_df.columns.values)


# In[ ]:


Errors = []

index = 0 
for prediction in validation_uncertainty_df["energy_demand_mean"]:
    Errors.append(mean_absolute_error([prediction], [np.float32(validation_df["Actual Val Output"].iloc[index])]))
    index+=1 

TrainingErrors = pd.DataFrame({"Date": validation_uncertainty_df["Date"], "Errors": Errors})
print(f"The average validation error: {np.mean(Errors)}")
print(f"The standard deviation of the validation error: ", np.std(Errors))


# In[ ]:


MonthErrors = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}

for i in range(len(validation_uncertainty_df["Date"])):
    date = validation_uncertainty_df["Date"].iloc[i]
    MonthErrors[date.month].append(Errors[i])
    
for key in MonthErrors.keys():
    mean = np.mean(MonthErrors[key])
    std = np.std(MonthErrors[key])
    MonthErrors[key] = (mean, std)
    print(f"Month {key}: Mean Error: {mean}, Standard deviation: {std}")


# In[ ]:


import matplotlib.pyplot as plt

# Assuming MonthErrors is a dictionary where:
# - keys are months
# - values are lists where the first element is the mean error and the second element is the standard deviation

# Extracting months, mean errors, and standard deviations
months = list(MonthErrors.keys())
mean_errors = [MonthErrors[key][0] for key in months]
std_devs = [MonthErrors[key][1] for key in months]

# Specifying that the lower part of the error bar is 0 and the upper part is the standard deviation
error_bars = np.array([(0,)*len(std_devs), std_devs])

# Creating the bar plot with error bars
plt.bar(months, mean_errors, yerr=error_bars, capsize=5)

plt.xlabel("Month")
plt.ylabel("Error in MWH")
plt.title("Monthly Error with Upper Standard Deviation")

plt.show()


# In[ ]:


#test_uncertainty_df.reset_index(inplace = True, drop = True)
Errors = []
index = 0 
for prediction in test_uncertainty_df["energy_demand_mean"]:
    Errors.append(mean_absolute_error([prediction], [np.float32(testing_df["Actual Test Output"].iloc[index])]))
    index+=1 

TrainingErrors = pd.DataFrame({"Date": test_uncertainty_df["Date"], "Errors": Errors})
print(f"The average validation error: {np.mean(Errors)}")
print(f"The standard deviation of the validation error: ", np.std(Errors))


# In[ ]:


MonthErrors = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}

for i in range(len(test_uncertainty_df["Date"])):
    date = test_uncertainty_df["Date"].iloc[i]
    MonthErrors[date.month].append(Errors[i])
    
for key in MonthErrors.keys():
    mean = np.mean(MonthErrors[key])
    std = np.std(MonthErrors[key])
    MonthErrors[key] = (mean, std)
    print(f"Month {key}: Mean Error: {mean}, Standard deviation: {std}")


# In[ ]:


import matplotlib.pyplot as plt

# Assuming MonthErrors is a dictionary where:
# - keys are months
# - values are lists where the first element is the mean error and the second element is the standard deviation

# Extracting months, mean errors, and standard deviations
months = list(MonthErrors.keys())
mean_errors = [MonthErrors[key][0] for key in months]
std_devs = [MonthErrors[key][1] for key in months]

# Specifying that the lower part of the error bar is 0 and the upper part is the standard deviation
error_bars = np.array([(0,)*len(std_devs), std_devs])

# Creating the bar plot with error bars
plt.bar(months, mean_errors, yerr=error_bars, capsize=5)

plt.xlabel("Month")
plt.ylabel("Error in MWH")
plt.title("Monthly Error with Upper Standard Deviation")

plt.show()


# # Generating the dataframe for mean, lower, and upper bounds

# In[ ]:


testing_df.drop(["Predicted Test Output"], axis = 1, inplace = True)
testing_df.insert(1, "99% lower_bound", test_uncertainty_df["99% lower_bound"])
testing_df.insert(2, "99.99% lower_bound", test_uncertainty_df["99.99% lower_bound"])
testing_df.insert(3, "99.999% lower_bound", test_uncertainty_df["99.999% lower_bound"])
testing_df.insert(4, "mean", test_uncertainty_df["energy_demand_mean"])
testing_df.insert(5, "99.999% upper_bound", test_uncertainty_df["99.999% upper_bound"])
testing_df.insert(6, "99.99% upper_bound", test_uncertainty_df["99.99% upper_bound"])
testing_df.insert(7, "99% upper_bound", test_uncertainty_df["99% upper_bound"])
testing_df.to_csv("TestingDataBounds2.csv", index = False)


# In[ ]:


validation_df.drop(["Predicted Val Output"], axis = 1, inplace = True)
validation_df.insert(1, "99% lower_bound", validation_uncertainty_df["99% lower_bound"])
validation_df.insert(2, "99.99% lower_bound", validation_uncertainty_df["99.99% lower_bound"])
validation_df.insert(3, "99.999% lower_bound", validation_uncertainty_df["99.999% lower_bound"])
validation_df.insert(4, "mean", validation_uncertainty_df["energy_demand_mean"])
validation_df.insert(5, "99.999% upper_bound", validation_uncertainty_df["99.999% upper_bound"])
validation_df.insert(6, "99.99% upper_bound", validation_uncertainty_df["99.99% upper_bound"])
validation_df.insert(7, "99% upper_bound", validation_uncertainty_df["99% upper_bound"])
validation_df.to_csv("ValidationDataBounds2.csv", index = False)


# In[ ]:


training_df.drop(["Predicted Train Output"], axis = 1, inplace = True)
training_df.insert(1, "99% lower_bound", training_uncertainty_df["99% lower_bound"])
training_df.insert(2, "99.99% lower_bound", training_uncertainty_df["99.99% lower_bound"])
training_df.insert(3, "99.999% lower_bound", training_uncertainty_df["99.999% lower_bound"])
training_df.insert(4, "mean", training_uncertainty_df["energy_demand_mean"])
training_df.insert(5, "99.999% upper_bound", training_uncertainty_df["99.999% upper_bound"])
training_df.insert(6, "99.99% upper_bound", training_uncertainty_df["99.99% upper_bound"])
training_df.insert(7, "99% upper_bound", training_uncertainty_df["99% upper_bound"])
training_df.to_csv("TrainingDataBounds2.csv", index = False)


# # Conclusions
# 
# - Bayesian LSTMs have been able to produce comparable performance to their frequentist counterparts (all else being equal)
# - Stochastic dropout enables users to approximate the posterior distribution of the target variable, \
# and thus construct confidence intervals for each prediction
# - Bayesian Neural Networks only attempt to account for epistemic model uncertainty and do not necessarily address aleatoric uncertainty
# - Computational overhead for repeated/multiple Bayesian LSTM predictions at inference to construct confidence intervals represent a potential challenge for real-time inference use-cases.
