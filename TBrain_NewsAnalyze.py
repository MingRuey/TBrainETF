"""
Usage: Analyze the news from TEJ database.
"""

import numpy
import pandas


def positive_wordbag():
    word = ['獲利穩', '回穩', '回溫', '看穩', '高檔', '走強', '增長', '看增',
            '倍增','季增', '月增', '年增','揚升', '新高', '成長', '看好',
            '看漲', '不俗','看旺', '賺']
    eval = ['優於大盤', '買進', '逢低買進', '強力買進', '加碼']

    return {'word':word, 'eval':eval}


def negative_wordbag():
    word = ['連虧', '衰退', '季減', '年減','月減', '低迷', '新低', '虧損', '減弱',
            '疲弱','走弱', '降溫', '大減', '熄火']
    eval = ['劣於大盤', '賣出', '減碼']
    return {'word':word, 'eval':eval}


def polarity(news):
    positive = positive_wordbag()
    negative = negative_wordbag()
    if '評等' in news:
        pos = any(words in news for words in positive['eval'])
        if pos:
            return 2
        else:
            neg = any(words in news for words in negative['eval'])
            return -2 if neg else 0
    else:
        pos = sum(i in news for i in positive['word'])
        neg = sum(i in news for i in negative['word'])
        return numpy.sign(pos-neg)


def main():
    df = '/archive/TBrain_ETF/NewsAnalyze/tb_news_utf8.csv'
    df = pandas.read_csv(df)

    # preprocessing
    df.columns = ['code', 'month-year', 'number', 'raw_date', 'title']
    df = df[['raw_date', 'code', 'title']]
    df['code'] = df['code'].str.split().apply(lambda x: x[0])

    t0 = pandas.to_datetime('20130102', format='%Y%m%d')
    df['date'] = (pandas.to_datetime(df['raw_date'], format='%Y-%m-%d') - t0).dt.days
    df = df[['date','raw_date','code', 'title']]

    # determine the polarity of news:
    df['polarity'] = df['title'].apply(polarity)
    df.to_csv('/archive/TBrain_ETF/tb_news_polarized.csv',
              index=False,
              encoding='utf-8'
              )


def news_by_etf():
    import os
    from tBrain_kfold import news_polarity

    news = '/archive/TBrain_ETF/NewsAnalyze/tb_news_polarized.csv'
    shares_map = '/archive/TBrain_ETF/Shares_to_ETF/SharesToEtfMap_on20180331_Tej.csv'
    news = news_polarity(news=news, shares_map=shares_map)

    path = '/archive/TBrain_ETF/taetf_t-series_0427/'
    files = os.listdir(path)
    for file in files:

        if file.endswith('.csv'):
            print(file)
            code = int(file.strip('taetf-code.csv'))

            date_range = (news.news['date'].min(), news.news['date'].max()+1)
            df = pandas.DataFrame({'date': range(*date_range)})

            t0 = pandas.to_datetime('20130102', format='%Y%m%d')
            df['raw_date'] = pandas.to_timedelta(df['date'], unit='D') + t0
            df['polarity'] = df['date'].apply(lambda date: news.get_polarity(etf_code=code, date_range=(date, date)))

            df.to_csv('/archive/TBrain_ETF/NewsAnalyze/taetf-code'+str(code)+'_withNews.csv', index=False)


if __name__ == '__main__':
    news_by_etf()
