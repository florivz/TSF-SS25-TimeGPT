import pandas as pd
import pmdarima as pm
import time
from prophet import Prophet
from nixtla import NixtlaClient
from dotenv import load_dotenv
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import timesfm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd

class Model:
    """
    Class:
    Put in a DataFrame with columns 'ts' (timestamp) and 'y' (target variable) when initializing.
    Splitting will be done in class.

    Methods:
    base_line() -> pd.DataFrame
        Baseline forecasting method (e.g., last value). Returns a DataFrame with columns 'ts' and 'yhat' for predictions.
    auto_arima() -> pd.DataFrame
        Forecasts using the auto_arima model from pmdarima. Returns a DataFrame with columns 'ts' and 'yhat' for predictions.
    LSTM() -> pd.DataFrame
        Forecasts using an LSTM neural network. Returns a DataFrame with columns 'ts' and 'yhat' for predictions.
    prophet() -> pd.DataFrame
        Forecasts using Facebook Prophet. Returns a DataFrame with columns 'ts' and 'yhat' for predictions.
    times_fm() -> pd.DataFrame
        Forecasts using TimesFM model. Returns a DataFrame with columns 'ts' and 'yhat' for predictions.
    time_gpt() -> pd.DataFrame
        Forecasts using the Nixtla TimeGPT model. Returns a DataFrame with columns 'ts' and 'yhat' for predictions.
    LSTM_no_window() -> pd.DataFrame
        Forecasts using an LSTM neural network without using DataWindow. Returns a DataFrame with columns 'ts' and 'yhat' for predictions.
    """


    def __init__(self, df, df_train=pd.DataFrame(), df_test=pd.DataFrame()):
        # Splitting manually to avoid index issues
        self.df = df.sort_values('ts').reset_index(drop=True)
        train_ratio = 0.8
        split_idx   = int(len(self.df) * train_ratio)

        if len(df_test)==0 and len(df_train)==0:
            self.df_train = self.df.iloc[:split_idx].copy().reset_index(drop=True)
            self.df_test  = self.df.iloc[split_idx:].copy().reset_index(drop=True)
        else:
            self.df_test = df_test
            self.df_train = df_train

        #self.health_check(self.df)


    def base_line(self) -> pd.DataFrame:
        length = len(self.df_test)
        last_value = self.df_train['y'][-length:].values
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': last_value})

    def auto_arima(self, m) -> pd.DataFrame:
        # ToDo Flo (auto arima from pmdarima)
        print("Starting ARIMA Training...\n")
        start_arima = time.time()
        y_train = self.df_train['y']

        model = pm.auto_arima(
            y_train,
            seasonal=True, m=m,
            d=1, D=1,                   
            test=None, seasonal_test=None,
            start_p=0, start_q=0, max_p=1, max_q=1,
            start_P=0, start_Q=0, max_P=1, max_Q=1,
            max_order=2,
            maxiter=4,
            stepwise=True,
            trace=False, 
            suppress_warnings=True
        )
        end_arima = time.time()
        duration_arima = end_arima - start_arima
        print("ARIMA Training Duration: ", duration_arima)

        fcst = model.predict(n_periods=len(self.df_test))
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': fcst.values}) 
    
    def LSTM(self, input_width=None, epoch=2) -> pd.DataFrame:
        df = self.df
        all_y = df["y"].values
        dataset=all_y.reshape(-1, 1)
        dataset = dataset[1:]
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # split into train and test sets, 50% test data, 50% training data
        train_size = int(len(dataset) * 0.8)
        orig = df.iloc[train_size:len(dataset),:]["ts"].copy().reset_index(drop=True)
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        # Use look_back as input_width if provided, else default to 240
        look_back = input_width or 240

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # Reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        # Create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(25, input_shape=(look_back, 1)))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(trainX, trainY, epochs=epoch, batch_size=240, verbose=1)

        # Make predictions
        testPredict = model.predict(testX)
        testPredict = scaler.inverse_transform(testPredict)

        ts_aligned = orig[look_back+1:look_back+1+len(testPredict)].reset_index(drop=True)
        result_df = pd.DataFrame({'ts': ts_aligned, 'yhat': testPredict.flatten()})
        return result_df

    
    def prophet(self) -> pd.DataFrame:
        df_train = self.df_train.rename(columns={"ts": "ds"})  # y bleibt y

        model = Prophet(
            daily_seasonality=False,  
            weekly_seasonality=True,
        )

        model.add_seasonality(
            name="intraday",
            period=1,         
            fourier_order=48    
        )

        start = time.time()
        model.fit(df_train)
        print(f"Prophet Training duration: {time.time()-start:.1f}s")

        future = pd.DataFrame({"ds": self.df_test["ts"]})

        fcst = model.predict(future)

        return pd.DataFrame({
            "ts":   fcst["ds"],     
            "yhat": fcst["yhat"]
        })

    
    def times_fm(self, freq, num_layers=50, checkpoint="google/timesfm-2.0-500m-pytorch", context_len=512, use_positional_embedding=False) -> pd.DataFrame:
        df_train_fm = self.df_train.copy()
        df_train_fm = df_train_fm.rename(columns={"ts": "ds", "y": "y"})
        df_train_fm["unique_id"] = "series_1"
        df_train_fm = df_train_fm[["unique_id", "ds", "y"]]

        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                per_core_batch_size=32,
                horizon_len=len(self.df_test),
                input_patch_len=32,
                output_patch_len=128,
                num_layers=num_layers,
                context_len=context_len,
                model_dims=1280,
                use_positional_embedding=use_positional_embedding
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=checkpoint),
                #huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )

        forecast_df = tfm.forecast_on_df(
            inputs=df_train_fm,
            freq=freq,       
            value_name="y", 
            num_jobs=-1,  
        )
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': forecast_df["timesfm"].values})
    
    # def time_gpt(self) -> pd.DataFrame:
    #     nixtla_train = self.df_train.copy()
    #     nixtla_train['unique_id'] = 'id1'
    #     nixtla_test = self.df_test.copy()
    #     nixtla_test['unique_id'] = 'id1'
        
    #     print("Nixtla DataFrame: ", nixtla_train.head())

    #     load_dotenv()
    #     nixtla_client = NixtlaClient(
    #         api_key=os.getenv('NIXTLA_API_KEY')
    #     )

    #     print(nixtla_client.validate_api_key())

    #     print("Starting TimeGPT Training...\n")
    #     start_gpt = time.time()
    #     timegpt_fcst_df = nixtla_client.forecast(
    #         df=nixtla_train,
    #         model='timegpt-1-long-horizon',
    #         id_col='unique_id',
    #         h=len(self.df_test),
    #         #freq='30min',
    #         time_col='ts',
    #         target_col='y',
    #         finetune_steps=10
    #     )
    #     end_gpt = time.time()
    #     period_gpt = end_gpt - start_gpt
    #     print("Nixtla Prediction Time: ", period_gpt)

    #     return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': timegpt_fcst_df['TimeGPT'].values}) 

    def time_gpt(self) -> pd.DataFrame:
        train_df = (self.df_train
                    .rename(columns={"ts": "ds", "y": "y"})
                    .assign(unique_id="series_1")
                    .sort_values("ds")
                    .reset_index(drop=True))

        load_dotenv()
        nixtla_client = NixtlaClient(api_key=os.getenv("NIXTLA_API_KEY"))

        horizon = len(self.df_test)                  
        fcst = nixtla_client.forecast(
            df=train_df,
            h=horizon,
            freq="30min",                            
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            model="timegpt-1",                       
            finetune_steps=100,                      
            finetune_depth=2,
            finetune_loss="mape"
        )

        return pd.DataFrame({
            "ts":   self.df_test["ts"],
            "yhat": fcst["TimeGPT"].values
        })



#-------------------------------------------------------
#--------------Helper Methods----------------------------
#-------------------------------------------------------
    def health_check(self, df, ts_col="ts", y_col="y"):
        df = self.df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        length      = len(df)
        duplicates  = df.duplicated(ts_col).sum()
        missing     = df[[ts_col, y_col]].isna().sum().sum()

        inferred = pd.infer_freq(df[ts_col])
        if inferred is None:                     
            inferred = df[ts_col].sort_values().diff().mode()[0]

        step       = inferred if isinstance(inferred, pd.Timedelta) else pd.Timedelta(inferred)
        irregular  = (df[ts_col].sort_values().diff().dropna() != step).sum()

        print(f"Len = {length} | duplicates = {duplicates}")
        print(f"Missing values (ts + y) = {missing}")
        print(f"Inferred frequency = {inferred}")
        print(f"Irregular {inferred} gaps = {irregular}")

# Datasets Flo: 1.1 & 1.3

# Datasets Ksenia: 1.2 & 1.4