from scipy import signal


# butterworth filter
def butter_filter(df, settings, use_lowpass=True, use_highpass=True, inplace=False):

    upper_threshold = settings.get("upper_threshold")
    lower_threshold = settings.get("lower_threshold")
    order = settings.get("order")
    columns = settings.get("columns")

    if not inplace:
        df = df.copy()
    for label in columns:
        if use_lowpass:
            b, a = signal.butter(order, upper_threshold, 'low', analog=False, fs=settings.get("sampling_frequency"))
            df[label] = signal.filtfilt(b, a, df[label])
        if use_highpass:
            d, c = signal.butter(order, lower_threshold, 'high', analog=False, fs=settings.get("sampling_frequency"))
            df[label] = signal.filtfilt(d, c, df[label])

    return df
