import pandas as pd
import scipy.stats as stats 

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

def get_hifi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', header = 0, index_col =0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    
    return hfi

#defining skewness fucntion: input is a series or datafram
def skewness(r):
    """
    Alternative to scipy.stats.skew()
    computes skewness of the supplied dataframe or series (returns float or series)
    """
    demeaned_r = r - r.mean()
    volatility = r.std(ddof=0)
    sigma_r = volatility
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3 


#defining Kurtosis fucntion: input is a series or datafram
def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    computes kurtosis of the supplied dataframe or series (returns float or series)
    """
    demeaned_r = r - r.mean()
    volatility = r.std(ddof=0)
    sigma_r = volatility
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4 

#Defing a is_normal fucniton: input is a series or datafram
def is_normal(r, level=0.01):
    '''
    Applies the Jarque-Bera test to determine if a series is normal (distributed) or not. Test is applied at the 1% level by default. Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = stats.jarque_bera(r)
    return p_value > level