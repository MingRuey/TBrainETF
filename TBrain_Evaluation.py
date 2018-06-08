"""

Compute the score of model using official evaluation rules.
First compute the score of each weak, average by total dates.

"""
import os
import pandas
import numpy
from tBrain_kfold import model_naiveavg, model_naivediff, model_naivenews


def days_to_date(df,
                 col_days='date',
                 t0=pandas.to_datetime('20130102', format='%Y%m%d')
                 ):
    """Inplace change the time delta column into datetime"""
    df[col_days] = pandas.to_timedelta(df[col_days], unit='D')+t0
    return None


def date_to_weekday(df, col_date='date', format='%Y%m%d'):
    """Inplace change the date column into weekday"""
    df[col_date] = pandas.to_datetime(df[col_date], format=format).dt.weekday
    return None


def weekday_to_weight(df, week_weight, col_date='date'):
    """Inplace map weekday to the given weight"""

    # somewhat ugly, need fix.
    mapping = {day: weight for day, weight in zip(range(7), week_weight)}
    df[col_date] = df[col_date].map(mapping)
    return None


def price_eval(df_model,
               df_etf,
               col_date='date',
               col_price='price',
               week_weight=[0.1, 0.15, 0.2, 0.25, 0.3, 0, 0]
               ):
    """Evaluate the price score with official metric"""
    weight = pandas.DataFrame({'date': df_model['date'].values})
    date_to_weekday(weight)
    weekday_to_weight(weight, week_weight=week_weight, col_date=col_date)

    # score series w/o weighted by weekdays
    score = (df_etf[col_price] - abs(df_model[col_price] - df_etf[col_price]))\
        / df_etf[col_price]

    # return weighted score
    return weight['date']*score


def trend_eval(df_model,
               df_etf,
               col_date='date',
               col_price='price',
               week_weight=[0.1, 0.15, 0.2, 0.25, 0.3, 0, 0]
               ):
    """Evaluate the trend score with official metric"""
    weight = pandas.DataFrame({'date': df_model['date'].values})
    date_to_weekday(weight)
    weekday_to_weight(weight, week_weight=week_weight, col_date=col_date)

    # score series w/o weighted by weekdays
    score = pandas.Series(
            numpy.where(df_model['price'] == df_etf['price'], 1, 0)
            )

    # return weighted score
    return score*weight['date']


def main_naiveavg():

    path = '/archive/TBrain_ETF/taetf_t-series_0427/'
    files = os.listdir(path)

    for file in files:
        if file.endswith('.csv'):
            df_etf = pandas.read_csv(path + file)
            df_model = model_naiveavg(df_etf, df_etf.shape[0]/7)
            df_etf = df_etf.drop(range(df_etf.shape[0]-df_model.shape[0]))

            days_to_date(df_model)
            days_to_date(df_etf)

            print(file)
            week_weight = [0.5, 0.75, 1, 1.25, 1.5, 0, 0]
            print(price_eval(df_model,
                             df_etf,
                             week_weight=week_weight
                             ).sum() / df_etf.shape[0]
                  )


def main_naivediff():

    path = '/archive/TBrain_ETF/taetf_t-series_0427/'
    files = os.listdir(path)

    for file in files:
        if file.endswith('.csv'):
            df_etf = pandas.read_csv(path + file)

            df_model = model_naivediff(df_etf, df_etf.shape[0] / 7)
            df_etf['price'] = numpy.where(
                    df_etf['price'].diff(periods=1) > 0, 1, 0
                    )

            df_etf = df_etf.drop(range(df_etf.shape[0] - df_model.shape[0]))

            days_to_date(df_model)
            days_to_date(df_etf)

            print(file)
            week_weight = [1, 1, 1, 1, 1, 1, 1]
            print(trend_eval(df_model,
                             df_etf,
                             week_weight=week_weight
                             ).sum() / df_etf.shape[0]
                  )


def main_naivenews():

    path = '/archive/TBrain_ETF/taetf_t-series_0427/'
    files = os.listdir(path)

    for file in files:

        if file.endswith('.csv'):
            print(file)
            code = int(file.strip('taetf-code.csv'))

            df_etf = pandas.read_csv(path + file)
            df_etf = df_etf[['date', 'price']]
            df_model = model_naivenews(df_etf,
                                       df_etf.shape[0] / 7,
                                       etf_code=code)
            df_etf['price'] = numpy.where(
                    df_etf['price'].diff(periods=1) > 0, 1, 0
                    )

            date_range = (729, 1941)

            df_etf = df_etf[(df_etf['date'] >= date_range[0]) &
                            (df_etf['date'] <= date_range[1])
                            ]
            df_model = df_model[(df_model['date'] >= date_range[0]) &
                                (df_model['date'] <= date_range[1])
                                ]

            days_to_date(df_model)
            days_to_date(df_etf)

            week_weight = [1, 1, 1, 1, 1, 1, 1]
            try:
                print(trend_eval(df_model,
                                 df_etf,
                                 week_weight=week_weight
                                 ).sum()/df_etf.shape[0]
                      )
            except ValueError as err:
                print(err)


if __name__ == "__main__":
    main_naivenews()
