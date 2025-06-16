import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('MaintenanceR.csv', encoding = 'utf-8', sep = ',')
data.head()

column_to_drop = ['UDI', 'Product ID']
for col in column_to_drop:
  data.drop(col, axis=1, inplace=True)

data.sample(10)

data.info()

data.describe()

data.nunique()

# А есть ли пустые значения?
# data['Temperature[C]'] = data['Temperature[C]'].fillna(data['eCO2[ppm]'].median())
# data['Humidity[%]'] = data['Humidity[%]'].fillna(data['eCO2[ppm]'].mean())
data.isna().sum()

data.isnull().sum()

# Преобразовывем к другим типам данных
for col in data.columns:
    if data[col].dtype == 'float64':  # Проверяем, является ли тип данных float
        if (data[col] % 1 == 0).all():  # Проверяем, все ли значения в столбце целые
            data[col] = data[col].astype(int)  # Приводим к целочисленному
            print(f"Столбец '{col}' был приведен к типу int.")
        else:
            print(f"Столбец '{col}' содержит нецелые значения и не может быть приведен к типу int.")
    else:
        print(f"Столбец '{col}' не является типом float и не требует приведения к int.")

  # Обогащение данных
data['Type'].value_counts()

data['Type'] = data['Type'].map({'L': 1, 'M': 2, 'H': 3})
data

# Добваляем новые термодинамические признаки
# ΔТемпература (delta_temp): Process temperature - Air temperature - gоказывает перегрев оборудования относительно среды. Высокие значения связаны с отказами.
# Относительный перегрев (rel_temp_ratio): (Process temperature - Air temperature) / Air temperature - нормированный показатель перегрева, учитывающий исходную температуру.
data['delta_temp'] = data['Process temperature [K]'] - data['Air temperature [K]']
data['rel_temp_ratio'] = data['delta_temp'] / data['Air temperature [K]']
data


# Проверяем дубликаты и если они есть, то удаляем
data = data.drop_duplicates()

data.duplicated().sum()

# Исследование выбросов
columns=['delta_temp', 'rel_temp_ratio', 'Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Machine failure','TWF',	'HDF',	'PWF',	'OSF',	'RNF']
data.describe()[columns]

# Ящик с усами
for column in columns:
  plt.figure()
  plt.title(f'Ящик с усами для {column}')
  plt.boxplot(data[column], vert=False)
  plt.show()


"""# Удаляем выбросы
# Верхние и нижние пределы выбросов
outlier = data[columns]
Q1 = outlier.quantile(0.1)
Q3 = outlier.quantile(0.9)

# IQR-оценки
IQR = Q3-Q1
data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]

index_list = list(data_filtered.index.values)
data_filtered = data[data.index.isin(index_list)]
data_filtered.sample(5)"""


"""for col in columns:
    plt.figure(figsize=(10, 6))
    plt.hist(data[col], alpha=0.5, label='С выбросами', color='red')
    plt.hist(data_filtered[col], label='Без выбросов', color='blue')
    plt.title(f'Гистограмма для {col}')
    plt.xlabel(col)
    plt.ylabel('Частота')
    plt.legend()
    plt.grid()
    plt.show()"""

# Диаграмма рассеяния
def plot_scatter(data, x, y):
    plt.figure(figsize=(8, 6))

    # Диаграмма рассеяния
    sns.regplot(x=x, y=y, data=data,
                scatter_kws={'s': 32, 'alpha': 0.8, 'color': 'skyblue'},
                line_kws={'color': 'red', 'linewidth': 2})

    plt.title(f'{x} vs {y}')
    plt.grid(alpha=0.2)
    plt.show()

# Примеры использования:
plot_scatter(data, 'Air temperature [K]', 'Process temperature [K]')
plot_scatter(data, 'Torque [Nm]', 'Tool wear [min]')
plot_scatter(data, 'Rotational speed [rpm]', 'Torque [Nm]')

# Круговая диаграмма
counts = data['Machine failure'].value_counts()
plt.figure()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Machine failure')
plt.show()

# Распределение гистограмма
data['Type'].plot(kind='hist', bins=20, title='Type')

# Корреляция
columns=['delta_temp', 'rel_temp_ratio', 'Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Machine failure','TWF',	'HDF',	'PWF',	'OSF',	'RNF']
plt.figure(figsize = (10,10))
df_selected = data[columns]
corr = df_selected.corr(method='spearman')
sns.heatmap(corr, annot=True, linewidths=1, cmap='coolwarm')
plt.title('Матрица корреляции')
plt.show()


"""
**Линейная регрессия с регуляризацией (Ridge, Lasso, ElasticNet)**
- Когда применять?
1. Данные имеют линейную зависимость между признаками и целевой переменной.
2. Присутствует мультиколлинеарность (коррелирующие признаки) → Ridge регрессия (L2-регуляризация).
3. Есть ненужные признаки → Lasso (L1-регуляризация) автоматически обнуляет некоторые веса.
4. Нужен баланс между L1 и L2 → ElasticNet.
5. Размер данных небольшой или средний (не подходит для очень больших данных, так как требует вычисления матриц).
- Что искать в EDA?
1. Проверить корреляцию признаков между собой (тепловая карта корреляции).
2. Посмотреть на распределение целевой переменной (если оно нормальное, линейная регрессия может работать хорошо).
3. Проверить линейность связи (графики sns.pairplot, sns.regplot).
3. Если признаки имеют разный масштаб → нужна стандартизация (StandardScaler).
**Решающие деревья (Decision Trees)**
- Когда применять?
1. Данные имеют нелинейные зависимости.
2. Есть категориальные признаки (деревья умеют работать без one-hot encoding).
3. Нужна интерпретируемость (можно визуализировать дерево).
4. Мало данных (случайный лес и бустинг могут переобучиться на малых данных).
5. Есть выбросы (деревья устойчивы к ним).
- Что искать в EDA?
1. Проверить нелинейность зависимостей (если линейные модели дают плохие результаты).
2. Посмотреть на выбросы (деревья их не боятся).
3. Если признаки имеют сложные взаимодействия (деревья могут их уловить).
**Случайный лес (Random Forest)**
- Когда применять?
1. Данные сложные, с нелинейными зависимостями.
2. Нужна устойчивость к переобучению (благодаря бэггингу).
3. Есть шум и выбросы (лес усредняет предсказания деревьев).
4. Размер данных средний или большой (но не гигантский).
- Что искать в EDA?
1. Если линейные модели плохо работают, а одно дерево переобучается → попробовать лес.
2. Если в данных есть много взаимодействий между признаками.
**Градиентный бустинг**
- Когда применять?
1. Данные большие и сложные (бустинг часто побеждает на соревнованиях).
2. Нужна максимальная точность (если правильно настроить).
3. Есть категориальные признаки.
4. Много пропусков и шума (бустинг устойчивее, чем случайный лес).
- Что искать в EDA?
1. Если случайный лес даёт хорошие результаты, но хочется лучше → пробуем бустинг.
2. Если в данных есть сложные нелинейные паттерны, которые не ловят другие модели.
**Стекинг (Stacking)**
- Когда применять?
1. Когда ансамбли (лес, бустинг) уже работают хорошо, но хочется выжать максимум.
2. Есть время и ресурсы для обучения нескольких моделей.
3. Размер данных не слишком большой (стэкинг требует много вычислений).
- Что искать в EDA?
Если разные модели дают разные ошибки (например, линейная модель ошибается на одних данных, а лес — на других), то стекинг может улучшить результат.
**Бэггинг (Bagging)**
- Когда применять?
1. Когда одна модель (например, дерево) склонна к переобучению.
2. Нужно уменьшить дисперсию ошибки (бэггинг усредняет предсказания).
3. Данные среднего размера (как и в случае случайного леса).
- Что искать в EDA?
Если одно дерево или другая простая модель сильно переобучается → бэггинг может помочь.

**Вывод: как по EDA выбрать модель?**
1. Проверить линейность:
Если зависимости линейные → пробуем линейную регрессию (+ регуляризацию, если много признаков).
Если нелинейные → деревья, лес, бустинг.
2. Размер данных:
Мало данных → деревья или линейные модели.
Много данных → случайный лес, бустинг.
3. Наличие выбросов:
Если много выбросов → деревья, лес, бустинг.
4. Категориальные признаки:
Если их много → градиентный бустинг или случайный лес.
5. Мультиколлинеарность:
Если признаки коррелируют → Ridge или ElasticNet.
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, BaggingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import joblib

# Выделяем целевую переменную и предикторы
df = data
label = 'Air temperature [K]'
features = ['delta_temp', 'rel_temp_ratio', 'Machine failure', 'Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','TWF',	'HDF',	'PWF',	'OSF',	'RNF']
X,y = df[features].values, df[label].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Применение PCA
from sklearn.decomposition import PCA

# Делаем, чтобы объяснить 95% дисперсии
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Исходное количество признаков: {X_train_scaled.shape[1]}")
print(f"Количество главных компонент после PCA: {X_train_pca.shape[1]}")


results_df = pd.DataFrame(columns=['Model', 'MSE', 'R2', 'MAE', 'RMSE', 'MAPE', 'Best Parameters'])

# Функция для добавления результатов модели
def add_results(model_name, y_true, y_pred, params=None):
    metrics = {
        'Model': model_name,
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'Best Parameters': str(params) if params else None
    }
    return metrics


# Регрессия с регуляризацией (Ridge, Lasso, ElasticNet)
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

params = {
    'Ridge': {'alpha': [0.01, 0.1]},
    'Lasso': {'alpha': [0.01, 0.1]},
    'ElasticNet': {'alpha': [0.01, 0.1], 'l1_ratio': [0.1, 0.5]}
}

for name, model in models.items():
    print(f"\n--- {name} ---")
    gs = GridSearchCV(model, params[name], cv=5, scoring='r2')
    gs.fit(X_train_pca, y_train)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test_pca)

    new_row = pd.DataFrame([add_results(name, y_test, y_pred, gs.best_params_)])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Метрики
    print(f"Лучшие параметры: {gs.best_params_}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

    # График
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{name} регрессия')
    plt.show()

# Дерево решений
tree_params = {
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree = DecisionTreeRegressor(random_state=42)
gs_tree = GridSearchCV(tree, tree_params, cv=5, scoring='r2')
gs_tree.fit(X_train_pca, y_train)  
best_tree = gs_tree.best_estimator_
y_pred = best_tree.predict(X_test_pca)

new_row = pd.DataFrame([add_results('Decision Tree', y_test, y_pred, gs_tree.best_params_)])
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Метрики
print(f"\n--- Decision Tree  ---")
print(f"Лучшие параметры: {gs_tree.best_params_}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

# График
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Дерево решений')
plt.show()


from sklearn.tree import export_text, export_graphviz
import graphviz
# Визуализация дерева
plt.figure(figsize=(20, 10))
dot_data = export_graphviz(
    best_tree,
    out_file=None,
    feature_names=[f"PC{i+1}" for i in range(X_train_pca.shape[1])],  # Названия компонент
    filled=True,
    rounded=True,
    special_characters=True,
    proportion=True,
    max_depth=3  # Ограничиваем глубину для лучшей читаемости
)
graph = graphviz.Source(dot_data)
display(graph) 

# Вывод текстовых правил (ограничим глубину для читаемости)
tree_rules = export_text(
    best_tree, 
    feature_names=[f"PC{i+1}" for i in range(X_train_pca.shape[1])],
    max_depth=3,
    decimals=2
)
print("\nПравила дерева (первые 3 уровня):")
print(tree_rules)

# Случайный лес
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)
gs_rf = GridSearchCV(rf, rf_params, cv=3, scoring='r2')
gs_rf.fit(X_train_pca, y_train) 
best_rf = gs_rf.best_estimator_
y_pred = best_rf.predict(X_test_pca)

new_row = pd.DataFrame([add_results('Random Forest', y_test, y_pred, gs_rf.best_params_)])
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Метрики
print(f"\n--- Random Forest ---")
print(f"Лучшие параметры: {gs_rf.best_params_}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

# График
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Случайный лес')
plt.show()


# Градиентный бустинг
gb_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

gb = GradientBoostingRegressor(random_state=42)
gs_gb = GridSearchCV(gb, gb_params, cv=3, scoring='r2')
gs_gb.fit(X_train_pca, y_train)
best_gb = gs_gb.best_estimator_
y_pred = best_gb.predict(X_test_pca)

new_row = pd.DataFrame([add_results('Gradient Boosting', y_test, y_pred, gs_gb.best_params_)])
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Метрики
print(f"\n--- Gradient Boosting ---")
print(f"Лучшие параметры: {gs_gb.best_params_}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

# График
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Градиентный бустинг')
plt.show()



# Стекинг
base_models = [
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('rf', RandomForestRegressor(random_state=42))
]

# Параметры для подбора базовых моделей
ridge_params = {'alpha': [0.01, 0.1]}
lasso_params = {'alpha': [0.001, 0.01]}
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [7, 15],
    'min_samples_split': [2, 5]
}

# Подбираем параметры для каждой базовой модели
ridge_gs = GridSearchCV(Ridge(), ridge_params, cv=3, scoring='r2')
ridge_gs.fit(X_train_pca, y_train) 
best_ridge = ridge_gs.best_estimator_

lasso_gs = GridSearchCV(Lasso(), lasso_params, cv=3, scoring='r2')
lasso_gs.fit(X_train_pca, y_train) 
best_lasso = lasso_gs.best_estimator_

rf_gs = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2')
rf_gs.fit(X_train_pca, y_train) 
best_rf = rf_gs.best_estimator_

# Обновляем список моделей с лучшими параметрами
base_models = [
    ('ridge', best_ridge),
    ('lasso', best_lasso),
    ('rf', best_rf)
]

# Параметры для финального оценщика
final_estimator_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# Создаем и настраиваем стекинг-регрессор
stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=GradientBoostingRegressor(random_state=42)
)

# Сетка параметров для стекинга
param_grid = {
    'final_estimator__n_estimators': final_estimator_params['n_estimators'],
    'final_estimator__learning_rate': final_estimator_params['learning_rate'],
    'final_estimator__max_depth': final_estimator_params['max_depth']
}

# Подбираем параметры для стекинга 
stacking_gs = GridSearchCV(
    estimator=stacking,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
stacking_gs.fit(X_train_pca, y_train)

# Лучшая модель
best_stacking = stacking_gs.best_estimator_
y_pred = best_stacking.predict(X_test_pca) 
new_row = pd.DataFrame([{
    'Model': 'Stacking (PCA)',
    'R2': r2_score(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
    'Best Parameters': str(stacking_gs.best_params_)
}])
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Выводим лучшие параметры
print("\n--- Stacking ---")
print("Лучшие параметры базовых моделей:")
print(f"Ridge: alpha={best_ridge.alpha}")
print(f"Lasso: alpha={best_lasso.alpha}")
print(f"Random Forest: n_estimators={best_rf.n_estimators}, max_depth={best_rf.max_depth}")

print("\nЛучшие параметры финального оценщика:")
print(f"n_estimators: {best_stacking.final_estimator_.n_estimators}")
print(f"learning_rate: {best_stacking.final_estimator_.learning_rate}")
print(f"max_depth: {best_stacking.final_estimator_.max_depth}")

# Метрики
print("\nМетрики на тестовой выборке:")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

# График
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Стекинг')
plt.grid(True)
plt.show()


# Бэггинг
# Параметры для GridSearch
bagging_params = {
    'n_estimators': [10, 20],
    'max_samples': [0.5, 0.7],
    'max_features': [0.5, 0.7]
}

# Базовый estimator
base_estimator = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

bagging = BaggingRegressor(
    estimator=base_estimator,
    random_state=42
)

# Поиск по сетке 
gs_bagging = GridSearchCV(bagging, bagging_params, cv=3, scoring='r2', n_jobs=-1, verbose=1)
gs_bagging.fit(X_train_pca, y_train) 

best_bagging = gs_bagging.best_estimator_
y_pred = best_bagging.predict(X_test_pca) 
new_row = pd.DataFrame([{
    'Model': 'Bagging',
    'R2': r2_score(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
    'Best Parameters': str(gs_bagging.best_params_)
}])
results_df = pd.concat([results_df, new_row], ignore_index=True)

print("\n--- Bagging ---")
print("Лучшие параметры модели:")
for param, value in gs_bagging.best_params_.items():
    print(f"{param}: {value}")

print("\nМетрики качества на тестовой выборке:")
metrics = {
    'R2': r2_score(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred)
}

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# График предсказанных vs истинных значений
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Бэггинг: Предсказанные vs Истинные значения')
plt.grid(True)
plt.show()


# Сводка метрик

results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)

print("Сводная таблица метрик всех моделей:")
display(results_df.style
       .format({
           'MSE': '{:.4f}',
           'R2': '{:.4f}',
           'MAE': '{:.4f}',
           'RMSE': '{:.4f}',
           'MAPE': '{:.2%}'
       }))


# Сохранение
joblib.dump(best_tree, 'tree_model.pkl')

# Сохраняем модель и преобразователи
import joblib
joblib.dump({'model': best_tree, 'pca': pca, 'scaler': scaler}, 'model.pkl')

# Загружаем модель
loaded = joblib.load('model.pkl')

# Подготовка данных для прогноза (пример)
new_data = np.array([[1.5, 0.8, 0, 310, 1500, 45, 0, 0, 0, 0, 0]])  # значения

# Делаем прогноз 
scaled_data = loaded['scaler'].transform(new_data)
pca_data = loaded['pca'].transform(scaled_data)
prediction = loaded['model'].predict(pca_data)[0]

print(f"Прогнозируемая температура: {prediction:.2f} K")
