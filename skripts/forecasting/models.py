import pandas as pd
import pmdarima as pm

class Model:
    df = pd.DataFrame()

    def __init__(self, df):
        self.df = df

    def base_line(self) -> pd.DataFrame:
        # ToDo: Ksenia (last value)
        return None

    def auto_arima(self, ts, seasonal=False, stepwise=True) -> pd.DataFrame:
        # ToDo Flo (auto arima from pmdarima)
        model = pm.auto_arima(
            self.df,
            seasonal=seasonal,
            stepwise=stepwise,     
            trace=True         
        )


        return None
    
    def LSTM(self) -> pd.DataFrame:
        # ToDo Ksenia
        return None
    
    def prophet(self) -> pd.DataFrame:
        # ToDo Flo
        return None
    
    def times_fm(self) -> pd.DataFrame:
        # ToDo Ksenia
        return None
    
    def time_gpt(self) -> pd.DataFrame:
        # ToDo Flo
        return None
    
# Datasets Flo: 1.1 & 1.3

# Datasets Ksenia: 1.2 & 1.4