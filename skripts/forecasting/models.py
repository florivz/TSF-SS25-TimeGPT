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
        load_dotenv()
        nixtla_client = NixtlaClient(
            api_key=os.getenv('NIXTLA_API_KEY')
        )

        print(nixtla_client.validate_api_key())

        print("Starting TimeGPT Training...\n")
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