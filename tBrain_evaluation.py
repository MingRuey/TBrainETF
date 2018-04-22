"""

Compute the score of model using official evaluation rules.
First compute the score of each weak, average by total dates.

"""

import pandas
import numpy
from tBrain_kfold import model_naiveavg
from tBrain_kfold import model_naivediff

def days_to_date(df, col_days='date', t0=pandas.to_datetime('20130102', format='%Y%m%d')):
    df[col_days] = pandas.to_timedelta(df[col_days], unit='D')+t0
    return None

# in place change the date column into weekday
def date_to_weekday(df, col_date='date', format='%Y%m%d'):
    df[col_date] = pandas.to_datetime(df[col_date], format=format).dt.weekday
    return None

def weekday_to_weight(df, week_weight, col_date='date'):
    df[col_date] = numpy.where(df[col_date] == 0, week_weight[0], \
                   numpy.where(df[col_date] == 1, week_weight[1], \
                   numpy.where(df[col_date] == 2, week_weight[2], \
                   numpy.where(df[col_date] == 3, week_weight[3], \
                   numpy.where(df[col_date] == 4, week_weight[4], \
                   numpy.where(df[col_date] == 5, week_weight[5], \
                   numpy.where(df[col_date] == 6, week_weight[6], None)))))))
    return None

def price_eval(df_model, df_etf, col_date='date', col_price='price', week_weight=[0.1, 0.15, 0.2, 0.25, 0.3, 0, 0]):
    weight =  df_model[[col_date, col_price]].copy()
    date_to_weekday(weight)
    weekday_to_weight(weight, week_weight=week_weight, col_date=col_date)

    # score series w/o weighted by weekdays
    eval = (df_etf[col_price] - abs(df_model[col_price] - df_etf[col_price]))/df_etf[col_price]

    # return weighted score
    return weight['date']*eval

def trend_eval(df_model, df_etf, col_date='date', col_price='price', week_weight=[0.1, 0.15, 0.2, 0.25, 0.3, 0, 0]):
    weight =  df_model[[col_date, col_price]].copy()
    date_to_weekday(weight)
    weekday_to_weight(weight, week_weight=week_weight, col_date=col_date)

    # score series w/o weighted by weekdays
    eval = pandas.Series(numpy.where(df_model['price']==df_etf['price'], 1, 0))

    # return weighted score
    return weight['date']*eval

def main_naiveavg():

    import os
    path = '/archive/TBrain_ETF/taetf_t-series/'
    files=os.listdir(path)

    for file in files:
        if file.endswith('.csv'):
            df_etf = pandas.read_csv(path + file)
            df_model = model_naiveavg(df_etf, df_etf.shape[0]/7)
            df_etf = df_etf.drop(range(df_etf.shape[0]-df_model.shape[0]))

            days_to_date(df_model)
            days_to_date(df_etf)

            print(file)
            week_weight = [0.5, 0.75, 1, 1.25, 1.5, 0, 0]
            print(price_eval(df_model, df_etf, week_weight=week_weight).sum()/df_etf.shape[0])

def main_naivediff():
    import os
    path = '/archive/TBrain_ETF/taetf_t-series/'
    files=os.listdir(path)

    for file in files:
        if file.endswith('.csv'):
            df_etf = pandas.read_csv(path + file)

            df_model = model_naivediff(df_etf, df_etf.shape[0] / 7)
            df_etf['price'] = numpy.where(df_etf['price'].diff(periods=1) > 0, 1, 0)

            df_etf = df_etf.drop(range(df_etf.shape[0] - df_model.shape[0]))

            days_to_date(df_model)
            days_to_date(df_etf)

            print(file)
            week_weight = [1, 1, 1, 1, 1, 1, 1]
            print(trend_eval(df_model, df_etf, week_weight=week_weight).sum()/df_etf.shape[0])

if __name__ == "__main__":
    main_naivediff()