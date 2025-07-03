import pandas as pd
import pmdarima as pm
import time
from prophet import Prophet
from nixtla import NixtlaClient
from dotenv import load_dotenv
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from window import DataWindow
from sklearn.model_selection import TimeSeriesSplit
import timesfm
from util import convert_time_stamp
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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

        self.health_check(self.df)


    def base_line(self) -> pd.DataFrame:
        length = len(self.df_test)
        last_value = self.df_train['y'][-length:].values
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': last_value})

    def auto_arima(self) -> pd.DataFrame:
        # ToDo Flo (auto arima from pmdarima)
        print("Starting ARIMA Training...\n")
        start_arima = time.time()
        y_train = self.df_train['y']

        model = pm.auto_arima(
            y_train,
            seasonal=True, m=48,
            d=1, D=1,                   
            test=None, seasonal_test=None,
            start_p=0, start_q=0, max_p=1, max_q=1,
            start_P=0, start_Q=0, max_P=1, max_Q=1,
            max_order=4,
            stepwise=True,
            trace=False, 
            suppress_warnings=True
        )
        end_arima = time.time()
        duration_arima = end_arima - start_arima
        print("ARIMA Training Duration: ", duration_arima)

        fcst = model.predict(n_periods=len(self.df_test))
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': fcst.values}) 

    
    def LSTM(self, input_width=24, label_width=1, shift=1) -> pd.DataFrame:
        df_train = self.df_train
        df_test = self.df_test
        df_train = convert_time_stamp(df_train.copy(), 'ts')
        df_test = convert_time_stamp(df_test.copy(), 'ts')

        split_index = int(len(df_train) * 0.8)
        train_df = df_train.iloc[:split_index]
        val_df = df_train.iloc[split_index:]

        scaler = MinMaxScaler()
        scaler.fit(train_df)

        train_df[train_df.columns] = scaler.transform(train_df)
        val_df[val_df.columns] = scaler.transform(val_df)
        df_test[df_test.columns] = scaler.transform(df_test)

        window = DataWindow(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df,
            val_df=val_df,
            test_df=df_test,
            label_columns=["y"]
        )

        train_ds = window.train
        val_ds = window.val
        test_ds = window.test

        feature_count = train_df.shape[1]

        model = Sequential([
            LSTM(32, return_sequences=False, input_shape=(input_width, feature_count)),
            Dense(label_width, kernel_initializer=tf.initializers.zeros)
        ])

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        early_stop = EarlyStopping(patience=3, monitor='val_loss', mode='min')

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[early_stop],
            verbose=2
        )

        y_pred_scaled = []
        for batch_x, _ in test_ds:
            batch_pred = model.predict(batch_x)
            y_pred_scaled.append(batch_pred)
        y_pred_scaled = np.concatenate(y_pred_scaled, axis=0)

        # Inverse transform predictions
        feature_columns = df_test.columns
        y_index = list(feature_columns).index("y")
        y_pred_full = np.zeros((y_pred_scaled.shape[0], len(feature_columns)))
        y_pred_full[:, y_index] = y_pred_scaled.flatten()
        y_pred_inverted = scaler.inverse_transform(y_pred_full)
        y_pred_original = y_pred_inverted[:, y_index]

        # Get correct timestamps for predictions
        ts_values = self.df_test['ts'].reset_index(drop=True)
        ts_values = ts_values[input_width + shift - 1:input_width + shift - 1 + len(y_pred_original)].reset_index(drop=True)

        return pd.DataFrame({
            'ts': ts_values,
            'yhat': y_pred_original
        })
    
    def prophet(self) -> pd.DataFrame:
        # Split into train and test using the same indices as before
        df_train_prophet = self.df_train.rename(columns={'ts': 'ds'})

        # Fit Prophet model
        prophet_model = Prophet(yearly_seasonality='auto', daily_seasonality='auto', weekly_seasonality='auto')

        start_prophet = time.time()
        prophet_model.fit(df_train_prophet)
        end_prophet = time.time()
        duration_prophet = end_prophet - start_prophet
        print('Prophet Training duration: ', duration_prophet)

        # Forecast
        future = prophet_model.make_future_dataframe(periods=len(self.df_test), freq='30min', include_history=False)
        fcst = prophet_model.predict(future)
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': fcst['yhat'].values}) 
    
    def times_fm(self, freq="D") -> pd.DataFrame:
        df_train_fm = self.df_train.copy()
        df_train_fm = df_train_fm.rename(columns={"ts": "ds", "y": "y"})
        df_train_fm["unique_id"] = "series_1"
        df_train_fm = df_train_fm[["unique_id", "ds", "y"]]

        # tscval = TimeSeriesSplit(n_splits=5, test_size=int(0.1 * len(df_train_fm)))
        # train_idx, test_idx = list(tscval.split(df_train_fm))[-1] # takes the last split
        # train_df, _ = df_train_fm.iloc[train_idx], df_train_fm.iloc[test_idx]

        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                per_core_batch_size=32,
                context_len=512,       
                horizon_len=len(self.df_test),       
                input_patch_len=32,    
                output_patch_len=128,  
                num_layers=50,         
                model_dims=1280,    
                use_positional_embedding=False
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

        forecast_df = tfm.forecast_on_df(
            inputs=df_train_fm,
            freq=freq,       
            value_name="y", 
            num_jobs=-1,  
        )

        forecast_df = forecast_df.iloc[:len(self.df_test['ts'])]

        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': forecast_df['timesfm'].values}) 
    
    def time_gpt(self) -> pd.DataFrame:
        nixtla_train = self.df_train.copy()
        nixtla_train['unique_id'] = 'id1'
        nixtla_test = self.df_test.copy()
        nixtla_test['unique_id'] = 'id1'
        
        print("Nixtla DataFrame: ", nixtla_train.head())

        load_dotenv()
        nixtla_client = NixtlaClient(
            api_key=os.getenv('NIXTLA_API_KEY')
        )

        print(nixtla_client.validate_api_key())

        print("Starting TimeGPT Training...\n")
        start_gpt = time.time()
        timegpt_fcst_df = nixtla_client.forecast(
            df=nixtla_train,
            model='timegpt-1-long-horizon',
            id_col='unique_id',
            h=len(self.df_test),
            freq='30min',
            time_col='ts',
            target_col='y',
            finetune_steps=10
        )
        end_gpt = time.time()
        period_gpt = end_gpt - start_gpt
        print("Nixtla Prediction Time: ", period_gpt)

        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': timegpt_fcst_df['TimeGPT'].values}) 

#-------------------------------------------------------
#--------------Helper Method----------------------------
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