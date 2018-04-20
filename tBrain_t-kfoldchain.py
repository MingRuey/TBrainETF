"""
Tools for chain forward prediction of a time series.
"""
import pandas
import numpy
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

    df = numpy.array([df['date'].values, df['price'].values]).transpose()

    kfold_indices = kfold_bytime(df[:,0], k)
    predict = numpy.zeros(df.shape)
    for i in kfold_indices[1:]:
        predict[i, 0] = df[i, 0]
        predict[i, 1] = numpy.average(df[i,1])

    predict = numpy.delete(predict, kfold_indices[0], axis=0) # remove the 1st fold
    return predict

def main():

    df = pandas.read_csv('/archive/TBrain_ETF/taetf_t-series/' + 'taetf-code50.csv')
    k = df.shape[0]/7

    predict = model_naiveavg(df, k)

    plt.plot(df['date'].values, df['price'].values, label='data')
    plt.plot(predict[:,0], predict[:,1], label='naive-k-fold')
    plt.xlabel('time in days')
    plt.ylabel('price')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()