import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Часть 1
# Задание 3
# Для датасета California Housing:
# 1) Разбейте датасет на тренировочную, валидационную и тестовую выборку
data_set = pd.read_csv('housing.csv')

# 2) Проведите преобразование категориального признака ocean_proximity через OneHot или Dummy-кодировку
data_set = pd.get_dummies(data_set, columns=['ocean_proximity']) #преобразую признак до разбиения датасета

train_validation, test_ds = train_test_split(data_set, test_size=0.1) #ds - data_set
train_ds, validation_ds = train_test_split(train_validation, test_size=0.1)

# 3) Замените признаки total_rooms и total_bedrooms на average_rooms и average_bedrooms (поделив на households).
train_ds["average_rooms"] = train_ds["total_rooms"] / train_ds["households"]
train_ds["average_bedrooms"] = train_ds["total_bedrooms"] / train_ds["households"]

validation_ds["average_rooms"] = validation_ds["total_rooms"] / validation_ds["households"]
validation_ds["average_bedrooms"] = validation_ds["total_bedrooms"] / validation_ds["households"]

test_ds["average_rooms"] = test_ds["total_rooms"] / test_ds["households"]
test_ds["average_bedrooms"] = test_ds["total_bedrooms"] / test_ds["households"]

train_ds = train_ds.drop(['total_rooms','total_bedrooms'], axis=1)
validation_ds = validation_ds.drop(['total_rooms','total_bedrooms'], axis=1)
test_ds = test_ds.drop(['total_rooms','total_bedrooms'], axis=1)


''' 4) В признаке average_bedrooms (total_bedrooms) есть отсутствующие значения.
Определите число экземпляров данных, для которых этот признак отсутствует.
Придумайте и обоснуйте стратегию заполнения пропусков в этой задаче. Заполните пропуски.
Заполняем средним количеством комнат. Таким образом количество комнат будет близко к вероятному.'''


print("Количество отсутствующих значений до заполнения:", data_set.isna().sum(), '\r\n')

mean_value = train_ds["average_bedrooms"].mean()
train_ds.fillna(value = mean_value, inplace=True)
test_ds.fillna(value = mean_value, inplace = True)
validation_ds.fillna(value = mean_value, inplace=True)

print("Количество отсутствующих значений после заполнения\n", train_ds.isna().sum(), validation_ds.isna().sum(), test_ds.isna().sum())

# 5) Нормализуйте признаки longitude и latitude
#   (сделайте так, чтобы каждый признак имел среднее значение 0 и дисперсию 1 внутри обучающей выборки)
Scaler = StandardScaler()
train_ds.loc[:, 'longitude':'latitude'] = Scaler.fit_transform(train_ds.loc[:, 'longitude':'latitude'].to_numpy())
test_ds.loc[:, 'longitude':'latitude'] = Scaler.transform(test_ds.loc[:, 'longitude':'latitude'].to_numpy())
validation_ds.loc[:, 'longitude':'latitude'] = Scaler.transform(validation_ds.loc[:, 'longitude':'latitude'].to_numpy())

print("Нормализация признаков longitude и latitude\n", train_ds, test_ds, validation_ds)
# Часть 2
# Задание 1
# Для датасета Davis:
# 1) Удалите некорректные данные

data_set = pd.read_csv('Davis.csv')

data_set.drop('Unnamed: 0', axis=1, inplace=True) #одногруппникам вы писали, что колонка индексов - лишняя
data_set.drop(["repwt", "repht"], axis=1) #эти колонки тоже вроде не нужно

data_set = data_set[np.logical_and(data_set['height'] > 150, data_set['weight'] < 100)]

# 2) Выделите тестовую выборку из 50 экземпляров
train_ds, test_ds = train_test_split(data_set, test_size=50)

# На тренировочных данных постройте:
#   1. Гистограмму height
#   2. Гистограмму weight
#   3. Эти же гистограммы для разных полов
figure = plt.figure(figsize=(10, 10), dpi=100)
ax = figure.add_subplot(1, 1, 1)

ax.hist(train_ds['height'], color='red', label='Height')
ax.hist(train_ds['weight'], color='blue', label='Weight')
ax.set_title('Weight and height')
ax.legend()
plt.show()

figure = plt.figure(figsize=(10, 10), dpi=100)
ax = figure.add_subplot(1, 1, 1)
ax.hist(train_ds[train_ds['sex'] == 'M']['height'], color='blue', label='Male Height')
ax.hist(train_ds[train_ds['sex'] == 'F']['height'], color='red', label='Female Height', alpha = 0.5)
ax.set_title('Weight for sex')
plt.show()

figure = plt.figure(figsize=(10, 10), dpi=100)
ax = figure.add_subplot(1, 1, 1)
ax.hist(train_ds[train_ds['sex'] == 'M']['weight'], color='blue', label='Male Weight')
ax.hist(train_ds[train_ds['sex'] == 'F']['weight'], color='red', label='Female Weight', alpha = 0.5)
ax.set_title('Height for sex')
ax.legend()
plt.show()

# 4) На тренировочных данных обучите классификатор пола (sex),
# используя только признаки height и weight. Замерьте производительность на тренировочной
# и тестовой выборке (через Accuracy). Рекомендуемые модели:  логистическая регрессия,
# quadratic discriminant analysis.

train_ds.replace({'M': 0, 'F': 1}, inplace=True)
test_ds.replace({'M': 0, 'F': 1}, inplace=True)

train_x = train_ds.loc[:, 'weight':'height'].to_numpy()
train_y = train_ds['sex'].to_numpy()

classifier = LogisticRegression().fit(train_x, train_y)
print('Производительность обучающей выборки:', classifier.score(train_x, train_y))

test_x = test_ds.loc[:, 'weight':'height'].to_numpy()
test_y = test_ds['sex'].to_numpy()
print('Производительность тестовой выборки:', classifier.score(test_x, test_y))

#5) Отобразите точки из обучающей выборки на плоскости (height-weight).
# Покрасьте их цветами в зависимости от пола. Раскрасьте области в зависимости
# от пола, предсказанного обученным в п.4. классификатором. Сделайте аналогичный график на тестовой выборке.

predicts = classifier.predict(train_x)

figure = plt.figure(figsize=(10, 10), dpi=100)
ax = figure.add_subplot(2, 2, 1)

x1_min, x1_max = train_x[:, 0].min() - 0.5, train_x[:, 1].max()+0.5
x2_min, x2_max = train_x[:, 0].min() - 0.5, train_x[:, 1].max()+0.5

xx1, xx2 = np.mgrid[x1_min:x1_max:150j, x2_min:x2_max:150j]
pred_x = np.column_stack([xx1.reshape(-1), xx2.reshape(-1)])
pred_y = classifier.predict(pred_x)

ax.scatter(train_x[predicts == 0][:, 0], train_x[predicts == 0][:, 1], color='blue', label='M')
ax.scatter(train_x[predicts == 1][:, 0], train_x[predicts == 1][:, 1], color='red', label='F')
ax.set_ylabel('height')
ax.set_xlabel('weight')
ax.set_title('Обучающая выборка')
ax.pcolormesh(xx1, xx2, pred_y.reshape(xx1.shape), cmap=ListedColormap(['blue', 'red']), alpha=0.5, shading='auto')
ax.set_xlim(40, 110)
ax.set_ylim(150, 190)
ax.legend()


x1_min, x1_max = test_x[:, 0].min() - 0.5, test_x[:, 1].max()+0.5
x2_min, x2_max = test_x[:, 0].min() - 0.5, test_x[:, 1].max()+0.5

xx1, xx2 = np.mgrid[x1_min:x1_max:150j, x2_min:x2_max:150j]
pred_x = np.column_stack([xx1.reshape(-1), xx2.reshape(-1)])
pred_y = classifier.predict(pred_x)


ax = figure.add_subplot(2, 2, 2)

predicts = classifier.predict(test_x)
ax.scatter(test_x[predicts == 0][:, 0], test_x[predicts == 0][:, 1], color='blue', label='M')
ax.scatter(test_x[predicts == 1][:, 0], test_x[predicts == 1][:, 1], color='red', label='F')
ax.set_ylabel('height')
ax.set_xlabel('weight')
ax.set_title('Тестовая выборка')
ax.pcolormesh(xx1, xx2, pred_y.reshape(xx1.shape), cmap=ListedColormap(['blue', 'red']), alpha=0.5, shading='auto')
ax.set_xlim(40,110)
ax.set_ylim(150, 190)
ax.legend()
plt.show()

# Задание 2.
#
# Для датасета CCPP.
# 1) Возьмите данные с листа 1 и выделите валидационную и тестовую выборку.

data_set = pd.read_excel('Folds5x2_pp.xlsx', sheet_name='Sheet1')

train_validation, test_ds = train_test_split(data_set, test_size=0.1)
train_ds, validation_ds = train_test_split(train_validation, test_size=0.1)

# 2) Постройте регрессионную модель.  Замерьте коэффициент R^2
# и среднюю ошибку предсказания на валидационной выборке.

train_y = train_ds["PE"].to_numpy()
train_x = train_ds.drop(["PE"], axis=1).to_numpy()

validation_y = validation_ds["PE"].to_numpy()
validation_x = validation_ds.drop(["PE"], axis=1).to_numpy()

test_y = test_ds["PE"].to_numpy()
test_x = test_ds.drop(["PE"], axis=1).to_numpy()

linear = LinearRegression().fit(train_x, train_y)
print(f'Коэффициент R^2: {linear.score(train_x, train_y)}')

prediction = linear.predict(validation_x)
print('Средняя ошибка предсказания:', math.sqrt(mean_squared_error(prediction, validation_y)))

# 3) Постройте точки из валидационной выборки на плоскости
# (t=истинное значение, y=предсказание модели). Отобразите вместе с ними прямую y=t.

plt.scatter(validation_y, prediction)
plt.plot(validation_y, validation_y, color='black')
plt.xlabel('Оценка')
plt.ylabel('Наблюдение')
plt.show()


