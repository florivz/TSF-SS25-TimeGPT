import pandas as pd
import pmdarima as pm
import time
from prophet import Prophet
from nixtla import NixtlaClient
from dotenv import load_dotenv
import os

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


    def __init__(self, df):
        # Splitting manually to avoid index issues
        self.df = df.sort_values('ts').reset_index(drop=True)
        train_ratio = 0.8
        split_idx   = int(len(self.df) * train_ratio)

        self.df_train = self.df.iloc[:split_idx].copy().reset_index(drop=True)
        self.df_test  = self.df.iloc[split_idx:].copy().reset_index(drop=True)

        self.health_check(self.df)


    def base_line(self) -> pd.DataFrame:
        # ToDo: Ksenia (last value)
        return None

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

    
    def LSTM(self) -> pd.DataFrame:
        # ToDo Ksenia
        return None
    
    def prophet(self) -> pd.DataFrame:
        # Split into train and test using the same indices as before
        df_train_prophet = self.df_train.rename(columns={'ts': 'ds'})

        # Fit Prophet model
        prophet_model = Prophet(yearly_seasonality='auto', daily_seasonality='auto', weekly_seasonality='auto')

        print("Starting Prophet Training...\n")
        start_prophet = time.time()
        prophet_model.fit(df_train_prophet)
        end_prophet = time.time()
        duration_prophet = end_prophet - start_prophet
        print('Prophet Training duration: ', duration_prophet)

        # Forecast
        future = prophet_model.make_future_dataframe(periods=len(self.df_test), freq='30min', include_history=False)
        fcst = prophet_model.predict(future)
        return pd.DataFrame({'ts': self.df_test['ts'], 'yhat': fcst['yhat'].values}) 
    
    def times_fm(self) -> pd.DataFrame:
        # ToDo Ksenia
        return None
    
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

        return pd.DataFrame({'ts': nixtla_test['ts'], 'yhat': timegpt_fcst_df['TimeGPT'].values}) 
    

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