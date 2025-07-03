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

        # DataFrame check
        dup_ct  = self.df.duplicated().sum()
        nan_ct  = self.df.isna().sum().sum()
        ts_diff = self.df['ts'].diff().dropna()

        if dup_ct:
            print(f"⚠️  Duplicates detected: {dup_ct}")
        if nan_ct:
            print(f"⚠️  NaN values detected: {nan_ct}")
        if ts_diff.nunique() > 1:
            print("⚠️  Irregular time intervals detected")

        # 30-Min frequency
        bad_intervals = ts_diff[ts_diff != pd.Timedelta('30min')]
        if not bad_intervals.empty:
            print("⚠️  Non-30-min gaps found:")
            print(bad_intervals.value_counts())
        else:
            print("✅ All timestamps are in a 30-minute grid")


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
        print(model.summary())
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

    
    def time_gpt(self) -> pd.DataFrame:
        load_dotenv()
        nixtla_client = NixtlaClient(
            api_key=os.getenv('NIXTLA_API_KEY')
        )

        print(nixtla_client.validate_api_key())

        start_gpt = time.time()
        timegpt_fcst_df = nixtla_client.forecast(
            df=self.df_train.reset_index(drop=True),
            h=len(self.df_test),
            freq='30min',
            time_col='ts',
            target_col='y'
        )
        end_gpt = time.time()
        period_gpt = end_gpt - start_gpt
        print("Nixtla Prediction Time: ", period_gpt)

        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': timegpt_fcst_df['TimeGPT'].values}) 
    
    

# Datasets Flo: 1.1 & 1.3

# Datasets Ksenia: 1.2 & 1.4