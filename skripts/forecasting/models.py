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
from window import DataWindow
import timesfm
from util import convert_time_stamp
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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

    
    def LSTM(self, input_width=None, label_width=None, shift=1, sequence_stride=1, auto_window=True) -> pd.DataFrame:
        orig = self.df_test['ts'].copy()
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
        df_train = convert_time_stamp(df_train, 'ts')
        df_test = convert_time_stamp(df_test, 'ts')

        # Auto windowing logic
        if auto_window:
            # Try to infer frequency and set window sizes accordingly
            n = len(df_train)
            # Heuristic: use 10% of train set as input window, 2% as label window, min 12/max 168
            input_width = input_width or max(12, min(168, n // 10))
            label_width = label_width or max(1, min(24, n // 50))
        else:
            input_width = input_width or 24
            label_width = label_width or 1

        split_index = int(len(df_train) * 0.8)
        train_df = df_train.iloc[:split_index].copy()
        val_df = df_train.iloc[split_index:].copy()

        scaler = MinMaxScaler()
        scaler.fit(train_df[['y']])
        train_df['y'] = scaler.transform(train_df[['y']])
        val_df['y'] = scaler.transform(val_df[['y']])
        df_test['y'] = scaler.transform(df_test[['y']])

        class FlexibleDataWindow(DataWindow):
            def make_dataset(self, data):
                data = np.array(data, dtype=np.float32)
                ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                    data=data,
                    targets=None,
                    sequence_length=self.total_window_size,
                    sequence_stride=sequence_stride,
                    shuffle=False,
                    batch_size=32
                )
                ds = ds.map(self.split_to_inputs_labels)
                return ds

        window = FlexibleDataWindow(
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

        # Prediction
        predictions = []
        for batch_inputs, _ in test_ds:
            batch_preds = model(batch_inputs)
            batch_preds = batch_preds.numpy().reshape(-1)
            predictions.extend(batch_preds)

        # Inverse transform only y
        inv_preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        # Align ts with the actual predictions (use self.df_test['ts'])
        ts_aligned = orig[-len(inv_preds):].reset_index(drop=True)
        return pd.DataFrame({'ts': ts_aligned, 'yhat': inv_preds})

    
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
    
    def times_fm(self, freq, num_layers=50, checkpoint="google/timesfm-2.0-500m-pytorch", context_len=512, use_positional_embedding=False) -> pd.DataFrame:
        df_train_fm = self.df_train.copy()
        df_train_fm = df_train_fm.rename(columns={"ts": "ds", "y": "y"})
        df_train_fm["unique_id"] = "series_1"
        df_train_fm = df_train_fm[["unique_id", "ds", "y"]]

        mean_y = df_train_fm['y'].mean()
        std_y = df_train_fm['y'].std()
        df_train_fm['y'] = (df_train_fm['y'] - mean_y) / std_y

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

        forecast_df["timesfm"] = (forecast_df["timesfm"] * std_y) + mean_y
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': forecast_df["timesfm"].values})
    
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