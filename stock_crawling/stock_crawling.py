import FinanceDataReader as fdr
import pandas as pd


ls = ['코스피', '코스닥', '네이버', '카카오', '삼성전자']
ls2 = ['KS11', 'KQ11', '035420', '035720', '005930']


def name_plus(x, y):
    ls = []
    for i in x:
        ls.append(y + '_' + i)

    return ls


df = pd.DataFrame()

for i, j in zip(ls, ls2):
    temp_df = fdr.DataReader(j, '2021')
    col = temp_df.columns.to_list()

    col = name_plus(col, i)
    temp_df.columns = col

    df = pd.concat([df, temp_df], axis=1)


df.to_csv('stock.csv', encoding='cp949')

