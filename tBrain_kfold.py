"""
Tools for chain forward prediction of a time series.
"""
import pandas
import matplotlib.pyplot as plt

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

def model_naiveavg(df,k):

    kfold_indices = kfold_bytime(df['date'].values, k)
    df_predict = pandas.DataFrame(0, index=range(df.shape[0]), columns=['date','price'])

    for i in range(len(kfold_indices)-1):
        df_predict.loc[kfold_indices[i+1], ['date']] = df.loc[kfold_indices[i+1], ['date']].values
        df_predict.loc[kfold_indices[i+1], ['price']] = df.loc[kfold_indices[i], ['price']].values.mean()

    # remove the 1st fold
    df_predict = df_predict.drop(kfold_indices[0])
    return df_predict

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

if __name__=='__main__':
    main_naivediff()