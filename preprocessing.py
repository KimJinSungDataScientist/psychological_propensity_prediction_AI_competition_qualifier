import pandas as pd
import numpy as np

train = pd.read_csv('./data/train.csv').drop('index',axis=1)
# train.drop('introelapse',axis=1,inplace=True)
# family size 이상치
train = train.drop(1019)
# age 이상치
train = train.drop(train[train['age']>100].index.to_list())
train = train.drop(train[train['familysize']>100].index.to_list())
# train = train.drop(train[train['introelapse']>3000].index.to_list())
# train = train.drop(train[train['testelapse']>10000].index.to_list())
# train = train.drop(train[train['surveyelapse']>10000].index.to_list()) # 나에 대한 추가 질문

test = pd.read_csv('./data/test.csv').drop('index',axis=1)
# test.drop('introelapse',axis=1,inplace=True)
test.loc[test['familysize']>100,'familysize']= train['familysize'].mean()
test.loc[test['age']>100,'age']= train['age'].mean()
# test.loc[test['introelapse']>3000,'introelapse']= train['introelapse'].mean()
# test.loc[test['testelapse']>10000,'testelapse']= train['testelapse'].mean()
# test.loc[test['surveyelapse']>10000,'surveyelapse']= train['surveyelapse'].mean()

test_index = pd.read_csv('./data/test.csv')['index']

pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 100)
value = train['country'].value_counts().values
rank = list(map(lambda x : 1 if x>1000 else (2 if x>100 else 3),value))
# rank = list(map(lambda x : 1 if x>1000 else (2 if x>500 else (3 if x>200 else (4 if x>100 else 5))),value))

temp_dict = {i : 0 for i in train['country'].value_counts().index.to_list()}

rank_dict = dict(zip(train['country'].value_counts().index.to_list(), rank))
rank_dict['nan'] = 0
train['country'] = train['country'].fillna('nan')
train['country'] = train['country'].apply(lambda x : rank_dict[x])
train['Ex'] = (train['TIPI1']+train['TIPI6'])/2
train['Ag'] = (train['TIPI7']+train['TIPI2'])/2
train['Con'] = (train['TIPI3']+train['TIPI8'])/2
train['Es'] =(train['TIPI9']+train['TIPI4'])/2
train['Op'] =(train['TIPI5']+train['TIPI10'])/2

train['mach_score'] = train[train.columns[:20]].apply(lambda x : x.mean(),axis=1)
train['T'] = train['Q1'] + train['Q2'] - train['Q3'] - train['Q6'] - train['Q7'] - train['Q10'] + train['Q12'] + train['Q15'] - train['Q16']
train['V'] = -train['Q4'] + train['Q5'] + train['Q8'] - train['Q11'] + train['Q13'] - train['Q14'] - train['Q17'] + train['Q18'] + train['Q20']
train['M'] = -train['Q9'] + train['Q19']
train['introelapse'] = np.log1p(train['introelapse'])
train['testelapse'] = np.log1p(train['testelapse'])
train['surveyelapse'] = np.log1p(train['surveyelapse'])
# train.drop('hand',axis=1,inplace=True)
# train.drop(['VCL7','VCL8','VCL11'],axis=1,inplace=True)

## 6,7,8,9,11,12 의미 없어 보임
# 6,8,10 실존하지 않는 단어 -> 얘네는 의미있게 작용하지 않을까?
# -> 7,8,11 제거

# train.drop([('TIPI'+str(i)) for i in range(1,11)],axis=1,inplace=True)
# train.drop([('Q'+str(i)) for i in range(1,20)],axis=1,inplace=True)

train['nature_score'] = train[[('Q'+str(i)) for i in range(20,27)]].apply(lambda x : x.mean(),axis=1)
# train.drop([('Q'+str(i)) for i in range(20,27)],axis=1,inplace=True)
# train.drop(['country','hand','introelapse','testelapse','surveyelapse'],axis=1,inplace=True)
train_fill_na = train.fillna(train.mean())


value = test['country'].value_counts().values
rank = list(map(lambda x : 1 if x>1000 else (2 if x>100 else 3),value))
# rank = list(map(lambda x : 1 if x>1000 else (2 if x>500 else (3 if x>200 else (4 if x>100 else 5))),value))

temp_dict = {i : 0 for i in test['country'].value_counts().index.to_list()}

rank_dict = dict(zip(test['country'].value_counts().index.to_list(), rank))
rank_dict['nan'] = 0
test['country'] = test['country'].fillna('nan')
test['country'] = test['country'].apply(lambda x : rank_dict[x])

test['Ex'] = (test['TIPI1']+test['TIPI6'])/2
test['Ag'] = (test['TIPI7']+test['TIPI2'])/2
test['Con'] = (test['TIPI3']+test['TIPI8'])/2
test['Es'] =(test['TIPI9']+test['TIPI4'])/2
test['Op'] =(test['TIPI5']+test['TIPI10'])/2

test['mach_score'] = test[test.columns[:20]].apply(lambda x : x.mean(),axis=1)
test['T'] = test['Q1'] + test['Q2'] - test['Q3'] - test['Q6'] - test['Q7'] - test['Q10'] + test['Q12'] + test['Q15'] - test['Q16']
test['V'] = -test['Q4'] + test['Q5'] + test['Q8'] - test['Q11'] + test['Q13'] - test['Q14'] - test['Q17'] + test['Q18'] + test['Q20']
test['M'] = -test['Q9'] + test['Q19']

test['introelapse'] = np.log1p(test['introelapse'])
test['testelapse'] = np.log1p(test['testelapse'])
test['surveyelapse'] = np.log1p(test['surveyelapse'])
# test.drop('hand',axis=1,inplace=True)
# test.drop(['VCL7','VCL8','VCL11'],axis=1,inplace=True)
# test.drop([('TIPI'+str(i)) for i in range(1,11)],axis=1,inplace=True)
# test.drop([('Q'+str(i)) for i in range(1,20)],axis=1,inplace=True)

test['nature_score'] = test[[('Q'+str(i)) for i in range(20,27)]].apply(lambda x : x.mean(),axis=1)
# test.drop(['country','hand','introelapse','testelapse','surveyelapse'],axis=1,inplace=True)
# test.drop([('Q'+str(i)) for i in range(20,27)],axis=1,inplace=True)
test_fill_na = test.fillna(test.mean())