# TSF-SS25-TimeGPT
Dedicated to the course TSF by Prof. Gernot Heisenberg @TH-Cologne

**Group A**
Members: Ksenia Blokhina, Florian Veitz

**Topic**
TimeGPT foundation model by Nixtla

1. Data Preperation
2. Model initialization
3. Model use
4. Evaluation & Recommendation

## Summary

<b>Baseline:</b> The forecast is a simple last value method, failing to capture trends or seasonality, resulting in large deviations from actual values.

<b>ARIMA:</b> Captures some seasonality and trend, but still misses changes.

<b>LSTM:</b> Follows the actual data closely, especially in capturing sudden changes.

<b>Prophet:</b> Good at modeling seasonality and trend, but can overfit or underfit in some periods.

<b>TimesFM and TimeGPT:</b> Both doesn't show tracking of the actual data


### Conclusion
<b>LSTMs</b> can model complex, nonlinear relationships in data, outperforming traditional linear models. It is evident that with enough data and proper regularization, LSTMs can learn to ignore noise and focus on underlying patterns. However, with limited and too noisy data, LSTM may fail to learn patterns.

Prophet and ARIMA models are good for interday data and also good in learning to ignore noise.

- In the egg sales case, the LSTM forecast is smoother than the actual, which is very noisy.

- In the electricity demand case, where the data doesn't have much noise the LSTM closely follows the up-and-down trend, while the ARIMA and Prophet models showed better results.

- In the saugeedday data case, the actual data has extreme spikes that the LSTM fails to fully predict.


## TimeGPT Evaluation

- **Multi-Step Forecast**: While predictions are fast with TimeGPT, overall performance was not good for long forecasting horizons, since things like Auto-Regression and Moving Averages are not learned directly.

Problem: Multi-Step Forecast will accumulate errors and context window exeeds limits easily.

- **Long Data History**:

Yearly seansonal patterns can only be captured by more then approx. 730 data points which often exeeds the TimeGPT context window.

Also their improved model 'timegpt-1-long-horizon' did not help much in our case.

- **Sensitivity to uncleaned data**: TimeGPT heavily relies on a well formatted and cleaned data. Therefore, it is not a very robust method. 

- **Fine Tuning**: Since the pre-trained modell already learnd from 100B+ data, it is hard to fine-tune this model even further. Even though with freezing bottom layers I could be difficult for the model to learn really new patterns.

In our case we took 100 steps for the fine-tuning which is a little above medium. This did not help too much since model might overfit when using more steps.

### Expectations
We expected that traditional approaches will outperform those foundation models, since the data was very clean and provided a lot of information.  The nearly anomaly-free, half-hourly series with strong cyclic structure rewards models that model seasonality and auto-regression explicitly.

### When to use TimeGPT
1. Easy and Fast implementation with not prior domain knowledge needed.
2. For small forecast horizons like 1-48 data points.
3. When having exogenous variables. This likely helps the model to understand the data
4. When data is sparse, TimeGPT performs better than traditional models.
5. When explainability is not important.

"<i>However, TimeGPT can be a great forecasting option for a broad audience that doesn’t have access to the specialized expertise required for building a custom model, especially if the forecasting horizon needed is short.<i>" - Claywr Apr. 2024 Time GPT for Forecasting: Kicking the Tires. **Medium**

### Key Takeaway
- LSTM, ARIMA and Prohpet when carefully implemented lead to very good results for both short term and long termn predictions
- Foundations models meant for any use case

**Requirements**
Python 3.10
Poetry

