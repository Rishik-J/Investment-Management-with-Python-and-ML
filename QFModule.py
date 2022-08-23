import pandas as pd
import numpy as np
import scipy.stats as stats 
from scipy.stats import norm

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

#Filtering returns less then 0 then computing std (using boolean mask)
def semidiviation(r):
    """
    Returns the semideviation of series r
    """
    is_negative = r<0
    return r[is_negative].std(ddof=0)

#defining skewness fucntion: input is a series or dataframe
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


#defining Kurtosis fucntion: input is a series or dataframe
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

#Defing a is_normal fucniton: input is a series or dataframe
def is_normal(r, level=0.01):
    '''
    Applies the Jarque-Bera test to determine if a series is normal (distributed) or not. Test is applied at the 1% level by default. Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = stats.jarque_bera(r)
    return p_value > level

#Defiing calculating historic VaR: input is a series or dataframe
def var_historic(r, level=5):
    """
    Returns the historic VaR at a specified level
    returns the number such that the "level" percent of the returns
    fall below that number
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a series or dataframe")

#Defining calculating Parametric VaR: input is a series or dataframe
def var_gaussian(r, level=5):
    """
    Returns the parametric Gaussian VaR of a Series or Dataframe
    """
    #Commpute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    return -(r.mean() + z * r.std(ddof=0))

#Defining calculating Cornish-Fisher VaR: input is a series or dataframe
def var_CornishFisher(r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a Series or Dataframe
    """
    #Commpute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 - 1)* s/6 +
                 (z ** 3 - 3*z)* (k-3)/24 -
                 (2*z**3 - 5*z)*(s**2)/36)
        
        
    return -(r.mean() + z * r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.DataFrame):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")