import numpy as np

def autocorrelation(series, max_lag=None):
    """
    Compute the autocorrelation function (ACF) for a given time series.
    
    Parameters:
    - series: 1D array-like, the input time series (e.g., particle positions)
    - max_lag: int, the maximum lag for which to calculate the ACF. 
               If None, it defaults to len(series) - 1.

    Returns:
    - lags: Array of lag values.
    - acf: Autocorrelation values corresponding to each lag.
    """
    n = len(series)
    mean = np.mean(series)
    variance = np.var(series)

    if max_lag is None:
        max_lag = n - 1

    # Compute the autocorrelation function for each lag
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        cov = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
        acf[lag] = cov / variance  # Normalize by the variance

    lags = np.arange(max_lag + 1)
    return lags, acf


# faster way of calculating autocorrelation via FFT
def autocorrelation_fft(series, max_lag=None):
    """
    Compute the autocorrelation function (ACF) using FFT for better performance.
    
    Parameters:
    - series: 1D array-like, the input time series (e.g., particle positions)
    - max_lag: int, the maximum lag for which to calculate the ACF. 
               If None, it defaults to len(series) - 1.

    Returns:
    - lags: Array of lag values.
    - acf: Autocorrelation values corresponding to each lag.
    """
    n = len(series)
    mean = np.mean(series)
    series = series - mean  # Remove the mean for zero-centered data

    # Use FFT to compute the autocorrelation
    fft_series = np.fft.fft(series, n=2*n)  # Zero-pad to 2*n for better FFT performance
    power_spectrum = np.abs(fft_series)**2
    acf = np.fft.ifft(power_spectrum).real[:n]  # Inverse FFT to get ACF

    # Normalize the ACF by the variance and set max_lag
    acf /= acf[0]  # Normalize by the zero-lag (variance)
    if max_lag is None:
        max_lag = n - 1

    lags = np.arange(max_lag + 1)
    return lags, acf[:max_lag + 1]
