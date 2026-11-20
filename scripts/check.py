import sys, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
p=sys.argv[1]
try:
    df=pd.read_csv(p)
    print("utf-8 OK")
except UnicodeDecodeError as e:
    print("utf-8 실패:", str(e)[:80]); df=pd.read_csv(p, encoding='cp1252'); print("cp1252 OK")
print(df.shape); print(df.columns.tolist()); print(df.isnull().sum().sum(), "결측치")
print(df.dtypes)
df.columns=['Date','Count','Hour','Temp','Humidity','Wind','Visibility','Dew','Solar','Rain','Snow','Seasons','Holiday','Functioning']
print(df['Functioning'].value_counts()); print("비운영일 대여합:", df[df.Functioning=='No'].Count.sum())
print(df.Seasons.value_counts().to_dict(), df.Holiday.value_counts().to_dict())
print(df.Count.describe())
print(df.groupby('Hour').Count.mean().round(0).to_dict())
print(df.groupby('Seasons').Count.mean().round(0).to_dict())
print(df.drop(columns=['Date','Seasons','Holiday','Functioning']).corr()['Count'].round(2).to_dict())
df=df[df.Functioning=='Yes'].drop(columns=['Functioning','Date'])
df['Seasons']=df['Seasons'].map({'Spring':0,'Summer':1,'Autumn':2,'Winter':3})
df['Holiday']=df['Holiday'].map({'No Holiday':0,'Holiday':1})
df['Commute']=df['Hour'].map(lambda h: 1 if h in [7,8,9,17,18,19] else 0)
df['Rainy']=(df['Rain']>0).astype(int)
sets={'단순: 기온':['Temp'],
 '다중1: +습도,강수,풍속':['Temp','Humidity','Rain','Wind'],
 '다중2: +시간,계절,공휴일':['Temp','Humidity','Rain','Wind','Hour','Seasons','Holiday'],
 '다중3: +출퇴근,비여부':['Temp','Humidity','Rain','Wind','Hour','Seasons','Holiday','Commute','Rainy'],
 '전체수치':['Temp','Humidity','Wind','Visibility','Dew','Solar','Rain','Snow','Hour','Seasons','Holiday','Commute','Rainy']}
for k,c in sets.items():
    X=df[c]; y=df['Count']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    m=LinearRegression().fit(Xtr,ytr); pr=m.predict(Xte)
    print(f"{k:28s} R2={r2_score(yte,pr):.3f} RMSE={mean_squared_error(yte,pr)**0.5:.0f} 음수예측={int((pr<0).sum())}")
    if k=='단순: 기온': print("  기온계수",round(m.coef_[0],1))
    if k.startswith('다중3'): print(dict(zip(c,np.round(m.coef_,1))))
