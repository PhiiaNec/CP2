import pandas as pd
import numpy as np
import itertools
import sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker

#подправим опции, которые позволят увеличить ширину рабочего поля IPython Notebook:
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

africa=pd.read_csv("tunisia.csv",sep=",")


'''
Постройте модель, которая оценивает вероятность того, что в следующем году в африканской стране Tunisia произойдет 
кризис на базе 5летней истории (наличие кризиса, информация о валюте и внешнем долге).


Изменим задание: используем колонки, где есть разные данные, а не одни нули:
systemic_crisis - Прогнозируемый показатель
sovereign_external_debt_default	
gdp_weighted_default
inflation_annual_cpi
currency_crises
inflation_crises
banking_crisis
'''
tunisia=africa[africa['country']=='Tunisia']
to_drop = ['case','cc3','country','year',"exch_usd","domestic_debt_in_default","independence"]
tunisia = tunisia.drop(to_drop,axis=1) #Удалим ненужные данные
tunisia=tunisia.replace({'no_crisis':"0",'crisis':"1"}) # Заменяем: no_crisis->0,crisis->1
print (tunisia.head(5)) #посмотрим первые 5 строк - все норм
'''
     systemic_crisis  sovereign_external_debt_default  gdp_weighted_default  inflation_annual_cpi  currency_crises  inflation_crises banking_crisis
822                0                                0                   0.0             22.013002                0                 1              0
823                0                                0                   0.0             24.582032                0                 1              0
824                0                                0                   0.0             24.996313                0                 1              0
825                0                                0                   0.0             72.109486                0                 1              0
826                0                                0                   0.0             37.613107                1                 1              0
       1              0
'''
scaler = StandardScaler() #Нормализуем данные
tunisia=scaler.fit_transform(tunisia)
x,y=[],[]  #Тут будут данные для прогноза и результат. Много вот таких структур:
n=0
for i in range(0,len(tunisia)-5,5):
    triple_set = []
    for k in range(i, i + 5):
        triple_set.extend(tunisia[k][1:])
    x.append(triple_set)
    y.append(tunisia[i+1][0])
'''
Вот так выглядят данные для обучения модели:
x=[[-0.320844473959874, -0.3208444739598739, 0................32738, 2.893959225697556, -0.2672612419124244], [-0.320844473959874, -0.3208....
в х Лежат массивы по 6 (кол-во столбцов для анализа * 5 лет = 30 элементов и таких массивов = кол-во лет /5
В y лежат результаты к каждому массиву х т.е. получаем структуру данных х[]  y[] ->   30шт[данные х] [данные в y (1шт)]
y=[-0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, 3.7416573867739413, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244, -0.2672612419124244]
'''
from sklearn.linear_model import LinearRegression # импортируем линейную регрессию
from sklearn.model_selection import train_test_split # импортируем случайную разбивалку данных на примерно пополам
x_train,x_test,y_train,y_test=train_test_split(x,y) # разбиваем данные, как повезет, получаем наборы x_train и y_train
# для обучения модели, x_test и y_test - вторые половинки исходного набора.
# В результате моделирования мы хотим получить данные, похожие на них
model= LinearRegression() #создаем объект модели из класса модели
model.fit(x_train,y_train) #Тренируем модель
predict_x_test=model.predict(x_test) #получаем из модели предсказание (скармливаем тестовый диапазон, получаем рассчет)

from sklearn.metrics import mean_absolute_error, mean_squared_error # либо, что бы оценить точность работы модели
print(f'mean_absolute_error={mean_absolute_error(y_test,predict_x_test)},mean_squared_error={mean_squared_error(y_test,predict_x_test)}')
'''
Бывает хороший результат
mean_absolute_error=0.008960066697390842,mean_squared_error=0.00015597491091363255   
Но раз на раз не приходится. Тунис не самый удобный пример, кризисы были несколько лет подряд 1 раз, поэтому модель 
не всегда учится на подходящих данных, или предсказание требует отрицательный ответ, что не сложно угадать.. 
на других странах работает веселее
'''

print(f'графики ожидаемого и предсказания')

import matplotlib. pyplot as plt   #Нарисуем графики ожидаемого и предсказания
df = pd.DataFrame({'Тестовая послед.':y_test,'предсказанн..':predict_x_test})
df.plot(color={'b','k'},linewidth="2")
df.plot().yaxis.set_major_locator(ticker.MultipleLocator(10))
df.plot().yaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.show()

'''
Сколько ввп в среднем у стран с наименьшим количеством кризисов в спокойные годы?
'''

countries=africa.groupby(["country"])["systemic_crisis"].sum().sort_values(ascending=True).head(15)

print(f'countries={countries}')
'''print(f'{africa.loc[(africa["country"].isin(countries)) & (africa["systemic_crisis"] == np.nan)]}')
'''
print('Сколько ввп в среднем у стран с наименьшим количеством кризисов в спокойные годы')

'''
Result:

Исходя из полученных данных, выделено 3 страны с наименьшим количеством кризисов. В среднем ввп по ним 0.


systemic_crisis  gdp_weighted_default
country                                            
Angola                      0              0.000000
Mauritius                   0              0.000000
South Africa                0              0.015789
Morocco                     2              0.010400
Algeria                     4              0.016235

'''

'''
В какой африканской стране кризисы происходят чаще?
'''
print(f'В какой африканской стране кризисы происходят чаще?')
print(f'{africa.groupby(["country"]).agg({"systemic_crisis":sum}).sort_values("systemic_crisis",ascending=False)}')
'''

согласно данным - в ЦАР.

1
                          systemic_crisis
country                                  
Central African Republic               19
Zimbabwe                               15
Kenya                                  13
Nigeria                                10
Egypt                                   6
Tunisia                                 5
Algeria                                 4
Ivory Coast                             4
Zambia                                  4
Morocco                                 2
Angola                                  0
Mauritius                               0
South Africa                            0
'''