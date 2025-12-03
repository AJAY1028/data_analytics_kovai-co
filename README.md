NAME OF DATASET: Daily_Public_Transport_Passenger_Journeys_by_Service_Type
DATE: 03/12/25, Wednesday

UNDERSTANDING THE DATASET:

Time-series dataset which represents transportation planning and urban economics.
It teaches me about how people are using a city's public transit network on a day-to-day basis.


INSIGHTS:

Bus Network Dominance: Rapid route and local route have dominated the other transportation. And even the Rapid Route is especially higher than the Local Route.

Peak Recovery Year: 2023 recorded the highest growth rate jump, marking the peak in ridership since the pandemic low.

Seasonality: The School, Peak Service and others columns frequently show zero counts on certain dates.

Variability: The system experiences extreme daily fluctuations, strongly indicating a huge difference between weekday/weekend or busy/holiday periods.

ALGORITHMS USED :
he SARIMA modeling process follows these general steps:

Stationarity Check (Integrated 'd' & 'D'): The model first checks if the data's mean and variance are constant over time. If not, it applies differencing (the 'd' and 'D' steps) until the series becomes stationary. This removes the overall trend and seasonality, leaving just the noise.

Parameter Identification (p, q, P, Q): Using diagnostic plots like the ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) , an analyst determines the optimal number of lags (p, q, P, Q) necessary to model the remaining noise.

Model Fitting: The parameters are input, and the model uses Maximum Likelihood Estimation (MLE) to find the coefficients that minimize the overall forecast error.

Forecasting: The fitted model is then used to generate future predictions



