# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:01:28 2018
@author: MRChou

An implementation of this paper by Jigar Patel 2015:
https://dl.acm.org/citation.cfm?id=2953241
"""

import numpy
import pandas


def ma(series, period=10, weight=None):
    """Return the moving average of the series,
       with first *aperiod rows become NaN"""
    if not weight:
        return series.rolling(window=period).mean()
    else:
        assert len(weight) == period
        out_series = pandas.Series(0, index=series.index)
        for i in range(period):
            out_series += series.shift(i) * weight[i]
        return out_series/sum(weight)


def momentum(series, period=10):
    """Return the differentiate series: S_t - S_t-avg_days+1,
       with first *avg_days rows become NaN"""
    return series - series.shift(period-1)


def rsi(series, period=10):
    """Return the relative strength index"""

    # get upward change U and downward change D, wiki: Relative strength index
    diff = series.diff(periods=1)
    up = numpy.where(diff > 0, diff, diff - diff)
    dw = numpy.where(diff < 0, diff, diff - diff)
    # diff - diff is a small trick to keep the NaN value in diff

    return 100 - 100 / (1 + ma(up, period=period) / ma(dw, period=period))


def exp_ma(series, n):
    """Return the exponential moving average of the series"""

    alpha = 2 / (n + 1)

    def recursive_ema(i):
        if i == (4*n-1):
            return alpha * series.shift(i)
        else:
            return alpha * series.shift(i) + (1-alpha) * recursive_ema(i+1)

    return recursive_ema(0)


def macd(series):
    """Ruturn the conventional 12-26-9 MACD"""
    return exp_ma(exp_ma(series, 12) - exp_ma(series, 26), 9)


def lowst_low_highest_high(low_series, high_series, period=10):
    """Reutrn the min-value series of *low_series withn the range of *period,
        and also the max-value series of *high_series."""
    ll = pandas.Series(float('inf'), index=low_series.index)
    hh = pandas.Series(float('-inf'), index=high_series.index)
    for i in range(period):
        ll = numpy.minimum(low_series.shift(i), ll)
        hh = numpy.maximum(high_series.shift(i), hh)
    return ll, hh


def stochastic_k(df, *,
                 col_price='price',
                 col_low='low',
                 col_high='high',
                 period=10):
    """Inplace add the Stochastic K% to the DataFrame"""
    ll, hh = lowst_low_highest_high(df[col_low], df[col_high], period=period)
    df['stochastic_k'] = df[col_price] - ll / (hh - ll)
    return None


def stochastic_d(df, stochastic_k_col='stochastic_k', period=10):
    """Inplace add Stochastic D% of the series from stochastic_k column"""
    df['stochastic_d'] = ma(df[stochastic_k_col], period=period)
    return None


def williams_r(df, *,
               col_price='price',
               col_low='low',
               col_high='high',
               period=10):
    """Inplace add the Williams %R series to the DataFrame"""
    ll, hh = lowst_low_highest_high(df[col_low], df[col_high], period=period)
    df['williams_r'] = (hh - df[col_price]) / (hh - ll)
    return None


def chaikin(df, *,
            col_price='price',
            col_low='low',
            col_high='high',
            col_vol='qunt'
            ):
    """Inplace add the Chaikin Oscillator to the DataFrame"""

    # the Accumulation Distribution Line
    multiplier = (2*df[col_price] - df[col_low] - df[col_high])\
        / (df[col_high] - df[col_low])
    money_flow = (df[col_vol] * multiplier).cumsum(skipna=True)
    df['chaikin'] = exp_ma(money_flow, 3) - exp_ma(money_flow, 10)
    return None


def cci(df, *,
        period=10,
        col_price='price',
        col_low='low',
        col_high='high'
        ):
    """Inplace add Commodity Channel Index to the DataFrame"""
    tp = (df[col_high] + df[col_low] + df[col_price])/3  # typical price
    matp = ma(tp, period=period)  # moving average of typical price
    md = (tp - matp).mad(skipna=True)  # mean deviation
    df['cci'] = (tp - matp) / (0.015 * md)
    return None


if __name__ == '__main__':
    pass