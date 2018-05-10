"""
Tools for chain forward prediction of a time series.
"""
import numpy
import pandas

# Split time series into k-fold
# Return the sorted indices of each fold
def kfold_bytime(tseries, k):

    # Get range of t, and split:
    foldsize = (tseries.max()-tseries.min())/k

    # Get the indice of which tseries is sorted
    indice_sort = sorted(range(len(tseries)), key=lambda s: tseries[s])

    # Split the indice into k-fold
    k_count = 1
    kfold_indices = []
    fold = []
    for i in indice_sort:
        if tseries[i] > k_count*foldsize:
            k_count += 1
            kfold_indices.append(fold)
            fold = []
        fold.append(i)
    kfold_indices.append(fold)

    return kfold_indices

# A simple k-fold averaging model to predict the price.
def model_naiveavg(df,k):

    kfold_indices = kfold_bytime(df['date'].values, k)
    df_predict = pandas.DataFrame(0, index=range(df.shape[0]), columns=['date','price'])

    for i in range(len(kfold_indices)-1):
        df_predict.loc[kfold_indices[i+1], ['date']] = df.loc[kfold_indices[i+1], ['date']].values
        df_predict.loc[kfold_indices[i+1], ['price']] = df.loc[kfold_indices[i], ['price']].values.mean()

    # remove the 1st fold
    df_predict = df_predict.drop(kfold_indices[0])
    return df_predict

# A simple k-fold difference model to predict the trend.
def model_naivediff(df, k):

    kfold_indices = kfold_bytime(df['date'].values, k)
    df_predict = pandas.DataFrame(0, index=range(df.shape[0]), columns=['date', 'price'])

    for i in range(len(kfold_indices)-1):
        val = df.loc[kfold_indices[i], ['price']].values
        if val.size:
            val = val[-1] - val[0]

            df_predict.loc[kfold_indices[i + 1], ['date']] = df.loc[kfold_indices[i + 1], ['date']].values
            df_predict.loc[kfold_indices[i + 1], ['price']] = int(val>0)

    # remove the 1st fold
    df_predict = df_predict.drop(kfold_indices[0])
    return df_predict


# Used by model_naivenews to get the polarity within certain range of date.
class news_polarity():

    def __init__(self, news, shares_map):
        self.news=pandas.read_csv(news)[['date','code','polarity']]
        self.shares_map= pandas.read_csv(shares_map)

    def get_polarity(self, etf_code, date_range):
        shares = self.shares_map[self.shares_map['etf_id']==etf_code]

        # limit news range
        news = self.news[(self.news['date'] >= date_range[0]) & (self.news['date'] <= date_range[1])]
        news = news[news['code'].isin(shares.stock_id.values)]

        # return the total polarity
        news['weight'] = news.code.apply(lambda code: shares[shares['stock_id']==code].percent.item())/100
        news['polarity'] = news['polarity']*news['weight']

        return news['polarity'].sum()

# A simple k-fold model considering the title of news from TEJ database.
def model_naivenews(df, k, etf_code, threshold=0.5):
    news = '/archive/TBrain_ETF/NewsAnalyze/tb_news_polarized.csv'
    shares_map = '/archive/TBrain_ETF/Shares_to_ETF/SharesToEtfMap_on20180331_Tej.csv'

    news = news_polarity(news=news, shares_map=shares_map)

    # Combine news with naivediff
    kfold_indices = kfold_bytime(df['date'].values, k)
    df_predict = pandas.DataFrame(0, index=range(df.shape[0]), columns=['date', 'price'])

    for i in range(len(kfold_indices)-1):
        val = df.loc[kfold_indices[i], ['price']].values
        if val.size:
            days = df.loc[kfold_indices[i + 1], ['date']].values
            df_predict.loc[kfold_indices[i + 1], ['date']] = days

            polar = news.get_polarity(etf_code, (int(days[0]), int(days[-1])))
            if polar:
                df_predict.loc[kfold_indices[i + 1], ['price']] = int(polar > threshold)
            else:
                df_predict.loc[kfold_indices[i + 1], ['price']] = int( (val[-1] - val[0])>0 )

    # remove the 1st fold
    df_predict = df_predict.drop(kfold_indices[0])
    return df_predict

import matplotlib.pyplot as plt
def main_naiveavg():

    df = pandas.read_csv('/archive/TBrain_ETF/taetf_t-series/' + 'taetf-code50.csv')
    k = df.shape[0]/7

    predict = model_naiveavg(df, k)

    plt.plot(df['date'].values, df['price'].values, label='data')
    plt.plot(predict['date'], predict['price'], label='naive-k-fold')
    plt.xlabel('time in days')
    plt.ylabel('price')
    plt.legend()
    plt.show()

def main_naivediff():

    df = pandas.read_csv('/archive/TBrain_ETF/taetf_t-series/' + 'taetf-code50.csv')
    k = df.shape[0]/7

    predict = model_naivediff(df, k)

    #df['price'] = numpy.where(df['price'].diff(periods=1) > 0, 1, 0)
    df=df.drop(range(df.shape[0]-predict.shape[0]))

    plt.scatter(df['date'], df['price'], c='b', label='data', s=10)
    plt.scatter(predict['date'], predict['price']*df['price'], c='r', label='naive-k-fold', s=2)
    plt.xlabel('time in days')
    plt.ylabel('price')
    plt.legend()
    plt.show()

def main_naivenews():

    file = "/archive/TBrain_ETF/taetf_t-series_0427/taetf-code50.csv"
    code = 50

    #date_range = pandas.read_csv('/archive/TBrain_ETF/NewsAnalyze/tb_news_polarized.csv')[['date']]
    #date_range = (date_range.min().item(), date_range.max().item())

    df_etf = pandas.read_csv(file)
    df_model = model_naivenews(df_etf, df_etf.shape[0] / 7, etf_code=code)
    df_etf['price'] = numpy.where(df_etf['price'].diff(periods=1) > 0, 1, 0)

    plt.scatter(df_etf['date'], df_etf['price'], c='b', label='data', s=10)
    plt.scatter(df_model['date'], df_model['price'], c='r', label='naive-news', s=2)
    plt.xlabel('time in days')
    plt.ylabel('price')
    plt.legend()
    plt.show()


if __name__=='__main__':
    main_naivenews()