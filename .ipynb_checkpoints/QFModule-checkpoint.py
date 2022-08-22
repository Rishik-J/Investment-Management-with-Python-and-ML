import pandas as pd

def drawdown(return_series: pd.Series):
    wealth_index = 1000 * (1+ return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth Index": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',                      header = 0,
                     index_col = 0,
                     na_values = -99.99)
    returns = me_m[['Lo 10', 'Hi 10']]
    returns.columns = ['Small cap', 'Large cap']
    returns = returns / 100
    returns.index = pd.to_datetime(returns.index, format = "%Y%m").to_period('M')
    return returns