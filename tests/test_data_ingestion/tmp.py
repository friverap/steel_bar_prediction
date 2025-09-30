from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='HO5W5PZ1IG7OYE7L', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
print(data.head(2))