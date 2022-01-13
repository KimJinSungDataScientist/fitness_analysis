import pandas as pd
import holidays
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

kr_holidays = holidays.KR()

df = pd.read_csv('./data/2020_통합_v2.0_my idea.csv',encoding='cp949')
df['휴일'] = df.Date.apply(lambda x: 'holiday' if x in kr_holidays else 'non-holiday')
df.head()
df.to_csv('./data/2020_통합_v2.0_my idea.csv', encoding='cp949', index=False)

# ### page 9 데이터 삭제


df = pd.read_csv('./data/2020_통합_v2.0_my idea.csv', encoding='cp949')

ls = set()
for dt, cnt in df.groupby('Date')['No'].count().items():
    if cnt <= 5:
        ls.add(dt)

mask = df['Date'].isin(ls)
df = df[~mask]

df.to_csv('./data/2020_통합_v2.1_my idea.csv', encoding='cp949', index=False)

# ### page 13 데이터 삭제

df = pd.read_csv('./data/2020_통합_v2.1_my idea.csv', encoding='cp949')
df = df[df['Gender2'] != '공용']
df.to_csv('./data/2020_통합_v2.2_my idea.csv', encoding='cp949', index=False)

# ### page 14 NULL 처리

df = pd.read_csv('./data/2020_통합_v2.2_my idea.csv', encoding='cp949')
df['Age2'] = df.apply(lambda x: None if x['Age'] <= 5 else x['Age2'], axis=1)
df.to_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949', index=False)

# ### page16 시각화

df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')
temp_df = pd.DataFrame(df.groupby('Area')['No'].count()).rename(columns={'No': '빈도'})
temp_df['퍼센트'] = np.round(temp_df['빈도'] / temp_df['빈도'].sum() * 100, 1)
temp_df['누적퍼센트'] = np.cumsum(temp_df['퍼센트'])
temp_df.loc['합계'] = np.sum(temp_df)[:-1]
print(temp_df.head())

# ### page 18 시각화
temp_df = pd.DataFrame(df.groupby('Month')['No'].count()).rename(columns={'No': '빈도'})
temp_df['퍼센트'] = np.round(temp_df['빈도'] / temp_df['빈도'].sum() * 100, 1)
temp_df['누적퍼센트'] = np.cumsum(temp_df['퍼센트'])
temp_df.loc['합계'] = np.sum(temp_df)[:-1]
temp_df
temp_df = df.groupby(['Month', 'Area'])['No'].count().unstack()
temp_df.loc['전체'] = temp_df.loc[1:12].sum().to_list()
temp_df['전체'] = [temp_df.iloc[i].sum() for i in range(0, 12)]
print(temp_df.head())

# ### page 24 시각화
for i in temp_df.index.values:
    temp_df.loc[str(i) + '%'] = np.round(temp_df.loc[i] / temp_df.loc['전체'] * 100, 1)
ls = []
for ind, i in enumerate(temp_df.index[:12]):
    ls.append(i)
    ls.append(temp_df.index[ind + 12])
temp_df = temp_df.reindex(index=ls)
print(temp_df.head())

# ### page 26 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')
temp_df = df.groupby(['Gender2', 'Month'])['No'].count().unstack()
temp_df.loc['전체'] = temp_df.iloc[:2].sum().to_list()
temp_df['전체'] = [temp_df.iloc[i].sum() for i in range(0, 3)]

for i in temp_df.index.values:
    temp_df.loc[str(i) + 'Gender2 중 %'] = np.round(temp_df.loc[i] / temp_df.loc[i]['전체'] * 100, 1)

for i in temp_df.index[:3].values:
    temp_df.loc[str(i) + 'Month 중 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체'] * 100, 1)

for i in temp_df.index[:3].values:
    temp_df.loc[str(i) + '전체 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체']['전체'] * 100, 1)

ls = []

for ind, i in enumerate(temp_df.index[:3]):
    ls.append(i)
    ls.append(temp_df.index[ind + 3])
    ls.append(temp_df.index[ind + 6])
    ls.append(temp_df.index[ind + 9])
temp_df = temp_df.reindex(index=ls)
print(temp_df.head())

# ### page 27 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')

temp_df = pd.DataFrame(df.groupby('Hour')['No'].count()).rename(columns={'No': '빈도'})
temp_df['퍼센트'] = np.round(temp_df['빈도'] / temp_df['빈도'].sum() * 100, 1)
temp_df['누적퍼센트'] = np.cumsum(temp_df['퍼센트'])
temp_df.loc['합계'] = np.sum(temp_df)[:-1]

print(temp_df.head())

# ### page 29 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')
temp_df = df.groupby(['요일', 'Hour'])['No'].count().unstack()

temp_df.loc['전체'] = temp_df.iloc[:7].sum().to_list()
temp_df['전체'] = [temp_df.iloc[i].sum() for i in range(0, 7)]

for i in temp_df.index.values:
    temp_df.loc[str(i) + ' 중 %'] = np.round(temp_df.loc[i] / temp_df.loc[i]['전체'] * 100, 1)

for i in temp_df.index[:7].values:
    temp_df.loc[str(i) + 'Hour 중 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체'] * 100, 1)

for i in temp_df.index[:7].values:
    temp_df.loc[str(i) + '전체 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체']['전체'] * 100, 1)

ls = []

for ind, i in enumerate(temp_df.index[:7]):
    ls.append(i)
    ls.append(temp_df.index[ind + 7])
    ls.append(temp_df.index[ind + 14])
    ls.append(temp_df.index[ind + 21])

print(temp_df.reindex(index=ls))

# ### page 31 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')

temp_df = pd.DataFrame(df.groupby('요일')['No'].count()).rename(columns={'No': '빈도'})
temp_df['퍼센트'] = np.round(temp_df['빈도'] / temp_df['빈도'].sum() * 100, 1)
temp_df['누적퍼센트'] = np.cumsum(temp_df['퍼센트'])
print(temp_df.head())

# ### page 33 시각화

df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')

temp_df = pd.DataFrame(df.groupby(['요일', 'Area'])['No'].count()).unstack()
temp_df.loc['전체'] = temp_df.iloc[0:7].sum().to_list()
temp_df['전체'] = [temp_df.iloc[i].sum() for i in range(0, 7)]

for i in temp_df.index.values:
    temp_df.loc[str(i) + '%'] = np.round(temp_df.loc[i] / temp_df.loc['전체'] * 100, 1)

ls = []

for ind, i in enumerate(temp_df.index[:7]):
    ls.append(i)
    ls.append(temp_df.index[ind + 7])

print(temp_df.reindex(index=ls))

# ### page 35 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')
temp_df = df.groupby(['Age2', 'Area'])['No'].count().unstack()

temp_df.boxplot(grid=False)
plt.axhline((temp_df.quantile(0.25).sum() / 8), color='r', linestyle='--', linewidth=1)
plt.axhline((temp_df.quantile(0.5).sum() / 8), color='r', linestyle='--', linewidth=1)
plt.axhline((temp_df.quantile(0.75).sum() / 8), color='r', linestyle='--', linewidth=1)

# ### page 37 시각화

df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')

temp_df = df.groupby(['Area', 'Age2'])['No'].count().unstack()
dict_temp = defaultdict(int)

for i in temp_df.columns:
    dict_temp[i // 10] += temp_df[i]

temp_df = pd.DataFrame(dict_temp)
temp_df.columns = (temp_df.columns * 10).to_list()

temp_df.loc['전체'] = temp_df.iloc[:8].sum().to_list()
temp_df['전체'] = [temp_df.iloc[i].sum() for i in range(0, 9)]

for i in temp_df.index.values:
    temp_df.loc[str(i) + ' Area 중 %'] = np.round(temp_df.loc[i] / temp_df.loc[i]['전체'] * 100, 1)

for i in temp_df.index[:9].values:
    temp_df.loc[str(i) + ' Age2Range 중 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체'] * 100, 1)

for i in temp_df.index[:9].values:
    temp_df.loc[str(i) + ' 전체 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체']['전체'] * 100, 1)

ls = []

for ind, i in enumerate(temp_df.index[:9]):
    ls.append(i)
    ls.append(temp_df.index[ind + 9])
    ls.append(temp_df.index[ind + 18])
    ls.append(temp_df.index[ind + 27])

print(temp_df.reindex(index=ls).rename(columns={0: 9.0}))

# ### page 41 시각화

df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')

temp_df = df.groupby(['Hour', 'Gender2'])['No'].count().unstack()
temp_df.loc['전체'] = temp_df.iloc[:20].sum().to_list()
temp_df['전체'] = [temp_df.iloc[i].sum() for i in range(0, 21)]

for i in temp_df.index.values:
    temp_df.loc[str(i) + '%'] = np.round(temp_df.loc[i] / temp_df.loc['전체'] * 100, 1)

ls = []

for ind, i in enumerate(temp_df.index[:21]):
    ls.append(i)
    ls.append(temp_df.index[ind + 21])

print(temp_df.reindex(index=ls))

# ### page 44 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')

plt.rc('font', family='Malgun Gothic')

temp_df = df.groupby(['Age2', 'Gender2'])['No'].count().unstack()
temp_df.boxplot()
plt.axhline((temp_df.quantile(0.5).sum() / 2), color='r', linestyle='--', linewidth=1)
plt.show()

# ### page 46 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')

temp_df = df.groupby(['Gender2', 'Area'])['No'].count().unstack()
temp_df.loc['전체'] = temp_df.iloc[:2].sum().to_list()
temp_df['전체'] = [temp_df.iloc[i].sum() for i in range(0, 3)]

for i in temp_df.index.values:
    temp_df.loc[str(i) + 'Gender2 중 %'] = np.round(temp_df.loc[i] / temp_df.loc[i]['전체'] * 100, 1)

for i in temp_df.index[:3].values:
    temp_df.loc[str(i) + 'Area 중 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체'] * 100, 1)

for i in temp_df.index[:3].values:
    temp_df.loc[str(i) + '전체 %'] = np.round(temp_df.loc[i] / temp_df.loc['전체']['전체'] * 100, 1)

ls = []

for ind, i in enumerate(temp_df.index[:3]):
    ls.append(i)
    ls.append(temp_df.index[ind + 3])
    ls.append(temp_df.index[ind + 6])
    ls.append(temp_df.index[ind + 9])

print(temp_df.reindex(index=ls))

# ### page 48 시각화
df = pd.read_csv('./data/2020_통합_v2.3_my idea.csv', encoding='cp949')
df = df[['Date', 'Area']]
weather = pd.read_csv('./data/2020_seoul_weather_v0.3.csv', encoding='cp949').rename(columns={'날짜': 'Date'})
weather = weather[['Date', '08시기준기온']]
temp_df = pd.merge(df, weather, how='inner', on='Date')
temp_df = temp_df[['Area', '08시기준기온']]

sns.heatmap(temp_df.groupby(['Area', '08시기준기온'])['Area'].count().unstack())