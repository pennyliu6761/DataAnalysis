# 6 探索性資料分析（EDA）
## 6.1 描述性統計分析：均值、中位數、標準差
### 6.1.1 描述性統計分析簡介
描述性統計分析是透過摘要和圖表來描述和總結資料的方法。它主要關注資料的集中趨勢和離散程度，幫助我們更好地理解資料的特性。

### 6.1.2 集中趨勢的度量

集中趨勢是指資料向中心值靠攏的程度，常用的度量包括均值和中位數。

#### 6.1.2.1 均值（Mean）

均值是資料集中所有數值的總和除以數值的個數。它反映了資料的平均水平。

* **公式：** 均值 = $(x_1 + x_2 + ... + x_n) / n$

```python
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5])
mean = data.mean()
print("均值：", mean)  # 輸出：3.0
```
#### 6.1.2.2 中位數（Median）
中位數是將資料按大小順序排列後位於中間位置的數值。當資料分佈偏斜或存在極端值時，中位數更能代表資料的典型值。

```python
import pandas as pd

data = pd.Series([1, 2, 3, 4, 100])
median = data.median()
print("中位數：", median)  # 輸出：3.0
```
### 6.1.3 離散程度的度量
離散程度是指資料分散的程度，常用的度量包括標準差。

#### 6.1.3.1 標準差（Standard Deviation）
標準差衡量資料偏離均值的程度。標準差越大，資料越分散；標準差越小，資料越集中。

* **公式：** 標準差 = $[Σ(x_i - 均值)^2 / (n - 1)]^{1/2}$
```python
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5])
std = data.std()
print("標準差：", std)  # 輸出：1.5811388300841898
```
### 6.1.4 實例分析：鳶尾花（Iris）資料集
我們使用經典的鳶尾花資料集來演示如何應用描述性統計分析。
```python
import pandas as pd
import seaborn as sns

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')
#sepal_length花萼長度  sepal_width花萼寬度  
#petal_length花瓣長度  petal_width花瓣寬度

# 描述性統計分析
print(iris.describe())

# 計算特定列的均值、中位數和標準差
print("\nsepal_length 的均值：", iris['sepal_length'].mean())
print("sepal_length 的中位數：", iris['sepal_length'].median())
print("sepal_length 的標準差：", iris['sepal_length'].std())
```
### 練習題
1. 讀取 titanic 檔案，並計算 Age 列的均值、中位數和標準差。
1. 讀取 titanic 檔案，並找出 Survived 為 1 的乘客中，Age 列的均值。
1. 建立一個包含 100 個隨機數的 Pandas Series，並計算其均值、中位數和標準差。
### 練習題解答
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 載入 Titanic 資料集
titanic = sns.load_dataset('titanic')

# 1. 計算 Age 列的均值、中位數和標準差
print("1. Age 均值：", titanic['age'].mean())
print("1. Age 中位數：", titanic['age'].median())
print("1. Age 標準差：", titanic['age'].std())

# Age 分布視覺化
plt.figure(figsize=(8, 5))
sns.histplot(titanic['age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 2. 找出 Survived 為 1 的乘客中，Age 列的均值
survived_age_mean = titanic[titanic['survived'] == 1]['age'].mean()
print("\n2. Survived 為 1 的乘客中，Age 均值：", survived_age_mean)

# 存活與未存活者的年齡分布箱型圖
plt.figure(figsize=(8, 5))
sns.boxplot(x='survived', y='age', hue='survived', legend=False, data=titanic, palette='Set2')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Age')
plt.show()

# 3. 建立隨機數 Series 並計算統計量
random_series = pd.Series(np.random.randn(100))
print("\n3. 隨機數 Series 均值：", random_series.mean())
print("3. 隨機數 Series 中位數：", random_series.median())
print("3. 隨機數 Series 標準差：", random_series.std())

# 隨機數 Series 的分布圖
plt.figure(figsize=(8, 5))
sns.histplot(random_series, bins=20, kde=True, color='lightcoral')
plt.title('Random Series Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```
* `seaborn.histplot()`參數說明
    |參數|意義|
    |-|-|
    |`titanic['age'].dropna()` | 要繪圖的資料（這裡是 `age` 欄位），並使用 `dropna()` 移除缺失值 `NaN`。|
    |`bins=30` | 將資料分成 30 個柱狀（bin），控制柱狀圖的寬度與數量。預設為 10。|
    |`kde=True` | 顯示 KDE（Kernel Density Estimation）核密度估計曲線，視覺化分布的平滑線。|
    |`color='skyblue'` | 設定柱狀圖的顏色為天藍色。你可以用任意合法的顏色名稱或 HEX 值（如 `#3498db`）。|

* `seaborn.boxplot()`參數說明
    |參數 | 說明|
    |-|-|
    |`x='survived'` | 設定 `X` 軸的分類欄位為 `survived`，即是否存活（0=未存活，1=存活）。每個類別會畫一個 box。|
    |`y='age'` | 設定 `Y` 軸的數值欄位為 `age`，也就是要畫箱型圖的數據來源。|
    |`hue='survived'`| 根據 `survived` 欄位再進行分組上色。雖然跟 `x='survived'` 一樣，但加上 `hue` 可用來指定 `palette` 的對應色。|
    |`legend=False` | 關閉圖例（`legend`）。如果不想顯示右上角的圖例就設為 `False`。|
    |`data=titanic` | 指定要使用的資料集是 `Titanic`。|
    |`palette='Set2'` | 設定色彩風格，`Set2` 是一組柔和的色系，適合類別分組視覺化。|

    ![image](https://hackmd.io/_uploads/ryCfmiNJel.png)
    ![image](https://hackmd.io/_uploads/SkYSXiV1le.png)
    ![image](https://hackmd.io/_uploads/HkiL7o4Jxg.png)



## 6.2 描述性統計分析：百分位數

### 6.2.1 百分位數簡介

百分位數將資料按大小順序排列，並指出有多少比例的資料小於或等於某個特定數值。例如，25% 百分位數表示有 25% 的資料小於或等於該數值。

### 6.2.2 百分位數的計算

Pandas 提供了 `quantile()` 函式來計算百分位數。該函式接受一個介於 0 和 1 之間的數值作為參數，表示要計算的百分位數。

```python
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
percentile_25 = data.quantile(0.25)
percentile_50 = data.quantile(0.5)
percentile_75 = data.quantile(0.75)
print("25% 百分位數：", percentile_25)
print("50% 百分位數：", percentile_50)
print("75% 百分位數：", percentile_75)

# 計算多個百分位數
percentiles = data.quantile([0.1, 0.5, 0.9])
print("\n10%、50% 和 90% 百分位數：\n", percentiles)
```
### 6.2.3 百分位數的應用
百分位數可以用於：

* 描述資料分佈： 百分位數可以幫助我們了解資料的分佈情況，例如，中位數（50% 百分位數）可以反映資料的中心位置。

```python
import pandas as pd
import seaborn as sns

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 計算 sepal_length 的中位數
median_sepal_length = iris['sepal_length'].median()
print("sepal_length 的中位數：", median_sepal_length)
```
* 檢測異常值： 百分位數可以用於檢測異常值，例如，我們可以將小於 5% 百分位數或大於 95% 百分位數的資料視為異常值。

```python
import pandas as pd
import seaborn as sns

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 計算 petal_width 的 5% 和 95% 百分位數
lower_bound = iris['petal_width'].quantile(0.05)
upper_bound = iris['petal_width'].quantile(0.95)

# 找出異常值
outliers = iris[(iris['petal_width'] < lower_bound) | (iris['petal_width'] > upper_bound)]
print("petal_width 的異常值：\n", outliers)
```
* 比較不同群組： 百分位數可以用於比較不同群組的資料分佈，例如，我們可以比較不同年齡組的收入百分位數。

```python
import pandas as pd

# 假設我們有一個包含年齡和收入的 DataFrame
data = {'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        'income': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]}
df = pd.DataFrame(data)

# 計算不同年齡組的收入中位數
median_income_by_age = df.groupby('age')['income'].median()
print("不同年齡組的收入中位數：\n", median_income_by_age)
```
* 評估相對表現： 例如，在考試成績中，我們可以知道某個學生的成績在所有考生中的相對位置。

```python
import pandas as pd

# 假設我們有一個包含學生姓名和考試成績的 DataFrame
data = {'student': ['A', 'B', 'C', 'D', 'E'],
        'score': [80, 90, 75, 85, 95]}
df = pd.DataFrame(data)

# 計算某個學生的成績百分位數
student_score = 85
percentile = (df['score'] < student_score).mean() * 100
print("學生成績的百分位數：", percentile)
```
### 6.2.4 實例分析：鳶尾花（Iris）資料集
我們使用經典的鳶尾花資料集來演示如何應用百分位數分析。
```python
import pandas as pd
import seaborn as sns

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 計算 sepal_length 的 25%、50% 和 75% 百分位數
sepal_length_percentiles = iris['sepal_length'].quantile([0.25, 0.5, 0.75])
print("sepal_length 的 25%、50% 和 75% 百分位數：\n", sepal_length_percentiles)

# 計算 petal_width 的 10%、30%、50%、70% 和 90% 百分位數
petal_width_percentiles = iris['petal_width'].quantile([0.1, 0.3, 0.5, 0.7, 0.9])
print("\npetal_width 的 10%、30%、50%、70% 和 90% 百分位數：\n", petal_width_percentiles)

# 找出 sepal_length 大於 75% 百分位數的資料
sepal_length_75_percentile = iris['sepal_length'].quantile(0.75)
sepal_length_above_75 = iris[iris['sepal_length'] > sepal_length_75_percentile]
print("\nsepal_length 大於 75% 百分位數的資料：\n", sepal_length_above_75.head())
```
![image](https://hackmd.io/_uploads/HJwmSjV1gx.png)



#### `matplotlib.pyplot`視覺化
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 計算 sepal_length 的 25%、50% 和 75% 百分位數
sepal_length_percentiles = iris['sepal_length'].quantile([0.25, 0.5, 0.75])
print("sepal_length 的 25%、50% 和 75% 百分位數：\n", sepal_length_percentiles)

# 視覺化 sepal_length 的分布與百分位數線
plt.figure(figsize=(10, 5))
sns.histplot(iris['sepal_length'], bins=20, kde=True, color='skyblue')
for quantile in sepal_length_percentiles:
    plt.axvline(quantile, color='red', linestyle='--')
plt.title('Sepal Length Distribution with 25%, 50%, 75% Quantiles')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# 計算 petal_width 的 10%、30%、50%、70% 和 90% 百分位數
petal_width_percentiles = iris['petal_width'].quantile([0.1, 0.3, 0.5, 0.7, 0.9])
print("\npetal_width 的 10%、30%、50%、70% 和 90% 百分位數：\n", petal_width_percentiles)

# 視覺化 petal_width 的分布與百分位數線
plt.figure(figsize=(10, 5))
sns.histplot(iris['petal_width'], bins=20, kde=True, color='lightgreen')
for quantile in petal_width_percentiles:
    plt.axvline(quantile, color='purple', linestyle='--')
plt.title('Petal Width Distribution with Selected Quantiles')
plt.xlabel('Petal Width')
plt.ylabel('Frequency')
plt.show()

# 找出 sepal_length 大於 75% 百分位數的資料
sepal_length_75_percentile = iris['sepal_length'].quantile(0.75)
sepal_length_above_75 = iris[iris['sepal_length'] > sepal_length_75_percentile]
print("\nsepal_length 大於 75% 百分位數的資料：\n", sepal_length_above_75.head())

# 視覺化篩選後的資料（以 sepal_length 對應 species 畫箱型圖）
plt.figure(figsize=(10, 5))
sns.boxplot(x='species', y='sepal_length', hue='species', data=sepal_length_above_75, palette='pastel')
plt.title('Sepal Length Distribution Above 75th Percentile by Species')
plt.show()
```

![image](https://hackmd.io/_uploads/HkF5HjNkxx.png)
![image](https://hackmd.io/_uploads/S1aoriVJlg.png)
![image](https://hackmd.io/_uploads/HJ1lLo4kxl.png)


#### `plotly.express`視覺化
```python
import pandas as pd
import seaborn as sns
import plotly.express as px
#import plotly.graph_objects as go

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 計算 sepal_length 的 25%、50%、75% 百分位數
sepal_length_percentiles = iris['sepal_length'].quantile([0.25, 0.5, 0.75])
print("sepal_length 的 25%、50% 和 75% 百分位數：\n", sepal_length_percentiles)

# 畫出 sepal_length 的分布圖（含百分位線）
fig1 = px.histogram(iris, x='sepal_length', nbins=20, title='Sepal Length 分布圖',
                    marginal='box', opacity=0.7, color_discrete_sequence=['skyblue'])

# 加上百分位線
for q in sepal_length_percentiles:
    fig1.add_vline(x=q, line_dash="dash", line_color="red")

fig1.update_layout(xaxis_title='Sepal Length', yaxis_title='次數', template='plotly_white')
fig1.show()

# 計算 petal_width 的 10%、30%、50%、70%、90% 百分位數
petal_width_percentiles = iris['petal_width'].quantile([0.1, 0.3, 0.5, 0.7, 0.9])
print("\npetal_width 的 10%、30%、50%、70% 和 90% 百分位數：\n", petal_width_percentiles)

# 畫出 petal_width 分布圖（含百分位線）
fig2 = px.histogram(iris, x='petal_width', nbins=20, title='Petal Width 分布圖',
                    marginal='violin', opacity=0.7, color_discrete_sequence=['lightgreen'])

for q in petal_width_percentiles:
    fig2.add_vline(x=q, line_dash="dash", line_color="purple")

fig2.update_layout(xaxis_title='Petal Width', yaxis_title='次數', template='plotly_white')
fig2.show()

# 找出 sepal_length > 75% 百分位數的資料
sepal_length_75_percentile = sepal_length_percentiles[0.75]
sepal_length_above_75 = iris[iris['sepal_length'] > sepal_length_75_percentile]
print("\nsepal_length 大於 75% 百分位數的資料：\n", sepal_length_above_75.head())

# 畫出這些篩選資料的箱型圖
fig3 = px.box(sepal_length_above_75, x='species', y='sepal_length', color='species',
              title='Sepal Length > 75th 百分位數 - 各 Species 箱型圖')
fig3.update_layout(template='plotly_white')
fig3.show()

```
![newplot](https://hackmd.io/_uploads/HJZi8jNJxe.png)
![newplot (1)](https://hackmd.io/_uploads/HyVQwsNylg.png)
![newplot (2)](https://hackmd.io/_uploads/r10VDsNkee.png)


### 6.2.5 練習題
1. 讀取 titanic 檔案，並計算 Age 列的 25%、50% 和 75% 百分位數。
1. 讀取 titanic檔案，並計算 Fare 列的 10%、30%、50%、70% 和 90% 百分位數。
1. 讀取 titanic檔案，並找出 Survived 為 1 的乘客中，Age 列的 25% 和 75% 百分位數。
1. 讀取 titanic 檔案，找出 Fare 大於 90% 百分位數的乘客資料。
1. 讀取 titanic 檔案，找出 Age 小於 10% 百分位數的乘客資料，並計算這些乘客的平均 Fare。
### 6.2.6 練習題解答
```python
import pandas as pd
import seaborn as sns

# 使用 seaborn 載入 Titanic 資料集
titanic = sns.load_dataset('titanic')

# 1. 計算 Age 列的 25%、50% 和 75% 百分位數
age_percentiles = titanic['age'].quantile([0.25, 0.5, 0.75])
print("1. Age 的 25%、50% 和 75% 百分位數：\n", age_percentiles)

# 2. 計算 Fare 列的 10%、30%、50%、70% 和 90% 百分位數
fare_percentiles = titanic['fare'].quantile([0.1, 0.3, 0.5, 0.7, 0.9])
print("\n2. Fare 的 10%、30%、50%、70% 和 90% 百分位數：\n", fare_percentiles)

# 3. 找出 Survived 為 1 的乘客中，Age 列的 25% 和 75% 百分位數
survived_age_percentiles = titanic[titanic['survived'] == 1]['age'].quantile([0.25, 0.75])
print("\n3. Survived 為 1 的乘客中，Age 的 25% 和 75% 百分位數：\n", survived_age_percentiles)

# 4. 找出 Fare 大於 90% 百分位數的乘客資料
fare_90_percentile = titanic['fare'].quantile(0.9)
fare_above_90 = titanic[titanic['fare'] > fare_90_percentile]
print("\n4. Fare 大於 90% 百分位數的乘客資料：\n", fare_above_90.head())

# 5. 讀取 `titanic.csv` 檔案，找出 `Age` 小於 10% 百分位數的乘客資料，並計算這些乘客的平均 `Fare`。
age_10_percentile = titanic['age'].quantile(0.1)
age_below_10 = titanic[titanic['age'] < age_10_percentile]
average_fare_below_10 = age_below_10['fare'].mean()
print(f"\n5. Age 小於 10% 百分位數的乘客平均 Fare：{average_fare_below_10}")
```
![image](https://hackmd.io/_uploads/BJG_YoVkgg.png)



## 6.3 相關性分析：皮爾森相關係數

### 6.3.1 相關性分析簡介

相關性分析是研究兩個或多個變數之間關係的方法。它可以幫助我們了解變數之間是否存在線性關係，以及關係的強弱和方向。

### 6.3.2 皮爾森相關係數

皮爾森相關係數（Pearson correlation coefficient）是一種衡量兩個連續變數之間線性關係的統計量。它的取值範圍在 -1 到 1 之間：

- 1：完全正相關
- -1：完全負相關
- 0：無線性相關

* **公式：** $r = Σ[(x_i - x̄)(y_i - ȳ)] / [√(Σ(x_i - x̄)²)√(Σ(y_i - ȳ)²)]$

其中，$x_i$ 和 $y_i$ 是兩個變數的觀測值，$x̄$ 和 $ȳ$ 是兩個變數的均值。

### 6.3.3 皮爾森相關係數的計算

Pandas 提供了 `corr()` 函式來計算皮爾森相關係數。

```python
import pandas as pd

data = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                     'B': [2, 3, 4, 5, 6],
                     'C': [5, 4, 3, 2, 1]})

# 計算皮爾森相關係數矩陣
correlation_matrix = data.corr()
print("皮爾森相關係數矩陣：\n", correlation_matrix)

# 計算特定列的相關係數
correlation_AB = data['A'].corr(data['B'])
print("\nA 列和 B 列的相關係數：", correlation_AB)
```
### 6.3.4 皮爾森相關係數的應用
皮爾森相關係數可以用於：
* 探索變數之間的關係： 了解哪些變數之間存在線性關係，以及關係的強弱和方向。

```python
import pandas as pd
import seaborn as sns

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 計算 sepal_length 和 petal_length 的皮爾森相關係數
correlation = iris['sepal_length'].corr(iris['petal_length'])
print("sepal_length 和 petal_length 的皮爾森相關係數：", correlation)

# 繪製散佈圖
sns.scatterplot(x='sepal_length', y='petal_length', data=iris)
```
![image](https://hackmd.io/_uploads/ryND2sNkle.png)

* 特徵選擇： 找出與目標變數相關性高的特徵，用於模型訓練。
```python
import pandas as pd
import seaborn as sns

# 載入鐵達尼號資料集（由 seaborn 提供）
titanic = sns.load_dataset('titanic')

# 計算與 survived 欄的皮爾森相關係數
correlation_with_survived = titanic.corr(numeric_only=True)['survived'].sort_values(ascending=False)
print("與 survived 欄的皮爾森相關係數：\n", correlation_with_survived)

# 選擇相關性大於 0.1 或小於 -0.1 的特徵
selected_features = correlation_with_survived[abs(correlation_with_survived) > 0.1].index.tolist()
print("\n選擇的特徵：", selected_features)
```
* 多重共線性檢測： 檢測自變數之間是否存在高度相關，避免模型出現多重共線性問題。
```python
import pandas as pd
import seaborn as sns

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 計算自變數之間的皮爾森相關係數矩陣
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
correlation_matrix = iris[features].corr()
print("自變數之間的皮爾森相關係數矩陣：\n", correlation_matrix)

# 繪製熱力圖
sns.heatmap(correlation_matrix, annot=True)
```
![image](https://hackmd.io/_uploads/BkYqniE1xe.png)


### 6.3.5 實例分析：鐵達尼（Titanic）資料集
我們使用經典的鐵達尼資料集來演示如何應用皮爾森相關係數分析。
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入 Titanic 資料集
titanic = sns.load_dataset('titanic')

# 計算數值欄位的皮爾森相關係數矩陣
correlation_matrix = titanic.corr(numeric_only=True)
print("皮爾森相關係數矩陣：\n", correlation_matrix)

# 繪製熱力圖
plt.figure(figsize=(10, 6))  # 設定圖表大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Titanic Dataset")
plt.show()
```
![image](https://hackmd.io/_uploads/Sy9B6sN1ge.png)

### 6.3.6 練習題
1. 讀取 titanic檔案，並計算 Age 和 Fare 列的皮爾森相關係數。
1. 讀取 titanic檔案，並找出與 Survived 列相關性最高的特徵。
1. 讀取 titanic檔案，並繪製所有數值列的皮爾森相關係數熱力圖。
1. 讀取 titanic檔案，找出 Pclass 和 Fare 的相關性，並解釋其意義。
### 6.3.7 練習題解答
```python
import pandas as pd
import seaborn as sns

# 載入 Titanic 資料集
titanic = pd.read_csv('titanic.csv')

# 1. 計算 Age 和 Fare 列的皮爾森相關係數
correlation_age_fare = titanic['Age'].corr(titanic['Fare'])
print("1. Age 和 Fare 的皮爾森相關係數：", correlation_age_fare)

# 2. 找出與 Survived 列相關性最高的特徵
correlation_with_survived = titanic.corr()['Survived'].sort_values(ascending=False)
print("\n2. 與 Survived 列的相關性：\n", correlation_with_survived)

# 3. 繪製皮爾森相關係數熱力圖
numerical_columns = titanic.select_dtypes(include=['number']).columns
correlation_matrix = titanic[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True)

# 4. 計算 Pclass 和 Fare 的相關性，並解釋其意義
correlation_pclass_fare = titanic['Pclass'].corr(titanic['Fare'])
print("\n4. Pclass 和 Fare 的皮爾森相關係數：", correlation_pclass_fare)
print("   意義：Pclass 和 Fare 之間存在負相關，表示船艙等級越高（Pclass 越小），船票價格（Fare）越高。")
```
## 6.4 資料分佈探索

### 6.4.1 資料分佈探索簡介

資料分佈探索是透過視覺化和統計方法來了解資料分佈情況的過程。它可以幫助我們識別資料的中心趨勢、變異性、偏度和峰度，以及是否存在異常值。

### 6.4.2 常用的資料分佈探索方法

* **直方圖（Histogram）：** 顯示資料的頻率分佈，幫助我們了解資料的集中趨勢和變異性。
* **箱形圖（Box Plot）：** 顯示資料的中位數、四分位數和異常值，幫助我們了解資料的變異性和偏度。
* **核密度估計圖（Kernel Density Plot）：** 平滑的直方圖，顯示資料的連續分佈。
* **小提琴圖（Violin Plot）：** 結合了箱形圖和核密度估計圖，顯示資料的變異性和分佈形狀。
* **QQ 圖（Quantile-Quantile Plot）：** 比較資料的分位數和理論分佈的分位數，幫助我們檢測資料是否符合特定的分佈。

### 6.4.3 使用 Python 進行資料分佈探索

我們可以使用 Matplotlib 和 Seaborn 函式庫來進行資料分佈探索。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 直方圖
plt.figure(figsize=(12, 12))
plt.subplot(3, 2, 1)
sns.histplot(iris['sepal_length'], kde=False)
plt.title('Sepal Length Histogram')

# 箱形圖
plt.subplot(3, 2, 2)
sns.boxplot(x='species', y='sepal_length', data=iris)
plt.title('Sepal Length Box Plot')

# 核密度估計圖
plt.subplot(3, 2, 3)
sns.kdeplot(iris['petal_width'])
plt.title('Petal Width Kernel Density Plot')

# 小提琴圖
plt.subplot(3, 2, 4)
sns.violinplot(x='species', y='petal_width', data=iris)
plt.title('Petal Width Violin Plot')

# QQ 圖
plt.subplot(3, 2, 5)
import scipy.stats as stats
stats.probplot(iris['sepal_width'], plot=plt)
plt.title('Sepal Width QQ Plot')

# 調整子圖間距（上、下、左、右、橫向間距、縱向間距）
plt.subplots_adjust(
    left=0.1, right=0.9,     # 圖的左右邊界（0 到 1 之間）
    top=0.9, bottom=0.1,     # 上下邊界
    wspace=0.3, hspace=0.4    # 子圖之間的間距（horizontal, vertical）
)

plt.show()
```
![image](https://hackmd.io/_uploads/SyJe6dUyll.png)

### 6.4.4 資料分佈探索的應用
了解資料的特性： 識別資料的中心趨勢、變異性和分佈形狀。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 年齡分佈直方圖
sns.histplot(titanic['age'].dropna(), kde=False)
plt.title('Age Distribution Histogram')
plt.show()
```
![image](https://hackmd.io/_uploads/SkH-a_8Jeg.png)


* 檢測異常值： 找出偏離資料分佈的異常值。
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 船票價格分佈箱形圖
sns.boxplot(x='pclass', y='fare', data=titanic)
plt.title('Fare Box Plot by Pclass')
plt.show()
```
![image](https://hackmd.io/_uploads/B1-zpOLkxe.png)


* 選擇合適的統計方法： 根據資料的分佈情況選擇合適的統計方法。
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 年齡分佈 QQ 圖
stats.probplot(titanic['age'].dropna(), plot=plt)
plt.title('Age QQ Plot')
plt.show()
```
![image](https://hackmd.io/_uploads/Sy5vkYU1lx.png)


* 特徵工程： 轉換資料分佈，使其更符合模型的要求。
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 船票價格分佈直方圖
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(titanic['fare'], kde=False)
plt.title('Fare Histogram (Original)')

# 對船票價格進行對數轉換
titanic['fare_log'] = np.log1p(titanic['fare'])

# 對數轉換後的船票價格分佈直方圖
plt.subplot(1, 2, 2)
sns.histplot(titanic['fare_log'], kde=False)
plt.title('Fare Histogram (Log Transformed)')
plt.show()
```
![image](https://hackmd.io/_uploads/rkPTyY81xe.png)

### 6.4.5 實例分析：鐵達尼號（Titanic）資料集
我們使用鐵達尼號資料集來演示如何進行資料分佈探索。
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 年齡分佈直方圖
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(titanic['age'].dropna(), kde=False)
plt.title('Age Distribution Histogram')

# 船票價格分佈箱形圖
plt.subplot(1, 3, 2)
sns.boxplot(x='pclass', y='fare', data=titanic)
plt.title('Fare Box Plot by Pclass')

# 性別分佈核密度估計圖
plt.subplot(1, 3, 3)
sns.kdeplot(titanic['age'][titanic['sex'] == 'male'].dropna(), label='Male')
sns.kdeplot(titanic['age'][titanic['sex'] == 'female'].dropna(), label='Female')
plt.title('Age Distribution by Sex')
plt.legend()
plt.show()
```
![image](https://hackmd.io/_uploads/SkSMgKUyel.png)

### 6.4.6 練習題
1. 讀取 titanic.csv 檔案，並繪製 Fare 列的直方圖和箱形圖。
1. 讀取 titanic.csv 檔案，並繪製 Age 列的核密度估計圖，按 Survived 列進行分組。
1. 讀取 titanic.csv 檔案，並繪製 Age 列的 QQ 圖。
1. 讀取 iris.csv 檔案，並繪製 sepal_length 列的小提琴圖，按 species 列進行分組。
### 6.4.7 練習題解答
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 1. 繪製 Fare 列的直方圖和箱形圖
plt.figure(figsize=(12, 12))
plt.subplot(2, 3, 1)
sns.histplot(titanic['fare'], kde=False)
plt.title('Fare Histogram')

plt.subplot(2, 3, 2)
sns.boxplot(titanic['fare'])
plt.title('Fare Box Plot')

# 2. 繪製 Age 列的核密度估計圖，按 Survived 列進行分組
plt.subplot(2, 3, 3)
sns.kdeplot(titanic['age'][titanic['survived'] == 0].dropna(), label='Not Survived')
sns.kdeplot(titanic['age'][titanic['survived'] == 1].dropna(), label='Survived')
plt.title('Age Distribution by Survival')
plt.legend()

# 3. 繪製 Age 列的 QQ 圖
plt.subplot(2, 3, 4)
stats.probplot(titanic['age'].dropna(), plot=plt)
plt.title('Age QQ Plot')

# 載入鳶尾花資料集
iris = sns.load_dataset('iris')

# 4. 繪製 sepal_length 列的小提琴圖，按 species 列進行分組
plt.subplot(2, 3, 5)
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.title('Sepal Length Violin Plot by Species')
plt.show()
```
![image](https://hackmd.io/_uploads/BymhxY8yll.png)

## 6.5 特徵探索

### 6.5.1 特徵探索簡介

特徵探索是透過分析資料的特徵（變數），來了解它們的特性、分佈、關係以及對目標變數的影響。它可以幫助我們選擇合適的特徵，進行特徵工程，以及構建更有效的機器學習模型。

### 6.5.2 特徵探索的方法

* **單變數探索：**
    * **數值型特徵：** 描述性統計（均值、中位數、標準差、百分位數）、直方圖、箱形圖、核密度估計圖、QQ 圖。
    * **類別型特徵：** 頻率統計、長條圖。
* **雙變數探索：**
    * **數值型 vs 數值型：** 散佈圖、皮爾森相關係數。
    * **類別型 vs 類別型：** 交叉表、卡方檢定。
    * **數值型 vs 類別型：** 箱形圖、小提琴圖、t 檢定、ANOVA。
* **多變數探索：**
    * **相關係數矩陣、熱力圖。**
    * **散佈圖矩陣、平行座標圖。**
    * **主成分分析（PCA）、t-SNE。**

### 6.5.3 使用 Python 進行特徵探索

我們可以使用 Pandas、Matplotlib 和 Seaborn 函式庫來進行特徵探索。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')
plt.figure(figsize=(12, 12))
# 單變數探索：數值型特徵
print(titanic['age'].describe())
plt.subplot(2, 2, 1)
sns.histplot(titanic['age'].dropna())

# 單變數探索：類別型特徵
print(titanic['sex'].value_counts())
plt.subplot(2, 2, 2)
sns.countplot(titanic['sex'])

# 雙變數探索：數值型 vs 數值型
plt.subplot(2, 2, 3)
sns.scatterplot(x='age', y='fare', data=titanic)
print(titanic['age'].corr(titanic['fare']))

# 雙變數探索：類別型 vs 類別型
print(pd.crosstab(titanic['survived'], titanic['pclass']))
print(stats.chi2_contingency(pd.crosstab(titanic['survived'], titanic['pclass'])))

# 雙變數探索：數值型 vs 類別型
plt.subplot(2, 2, 4)
sns.boxplot(x='pclass', y='age', data=titanic)
plt.show()
```
![image](https://hackmd.io/_uploads/HJ1RWtUkel.png)

### 6.5.4 特徵探索的應用
* 理解資料的特性： 識別特徵的分佈、範圍、異常值和缺失值。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 年齡分佈直方圖
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(titanic['age'].dropna())
plt.title('Age Distribution')

# 船票價格箱形圖
plt.subplot(1, 2, 2)
sns.boxplot(y='fare', data=titanic)
plt.title('Fare Boxplot')
plt.show()
```
![image](https://hackmd.io/_uploads/B1JuMYIJex.png)

* 選擇合適的特徵： 找出與目標變數相關性高的特徵，以及冗餘或無用的特徵。

```python
import pandas as pd
import seaborn as sns

# 載入資料
titanic = sns.load_dataset('titanic')

# 只選擇數值欄位來計算相關性
numeric_df = titanic.select_dtypes(include=['number'])

# 計算與 survived 的相關性
correlation = numeric_df.corr()['survived'].abs().sort_values(ascending=False)
print(correlation)

# 選擇相關性高的特徵
selected_features = correlation[correlation > 0.1].index.tolist()
print(f'Selected Features: {selected_features}')
```
* 特徵工程： 轉換、組合或衍生特徵，以提高模型的性能。
```python
import pandas as pd
import numpy as np

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 創建 AgeGroup 特徵
bins = [0, 12, 18, 60, np.inf]
labels = ['Child', 'Teenager', 'Adult', 'Senior']
titanic['AgeGroup'] = pd.cut(titanic['age'], bins=bins, labels=labels)

# 創建 FamilySize 特徵
titanic['FamilySize'] = titanic['sibsp'] + titanic['parch'] + 1
print(titanic)
```
* 模型解釋： 了解特徵對模型的影響，以及模型的預測結果。
```python
import pandas as pd
import statsmodels.formula.api as smf

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 使用邏輯迴歸模型
model = smf.logit('survived ~ pclass + sex + age', data=titanic).fit()
print(model.summary())
```
### 6.5.5 實例分析：鐵達尼號（Titanic）資料集
我們使用鐵達尼號資料集來演示如何進行特徵探索。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鐵達尼號資料集
titanic = sns.load_dataset('titanic')

# 性別與存活率的關係
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.countplot(x='sex', hue='survived', data=titanic)

# 艙等與存活率的關係
plt.subplot(1, 3, 2)
sns.countplot(x='pclass', hue='survived', data=titanic)

# 年齡與存活率的關係
plt.subplot(1, 3, 3)
sns.violinplot(x='survived', y='age', data=titanic)
plt.show()
```
![image](https://hackmd.io/_uploads/HkldQFIygg.png)

### 6.5.6 練習題
1. 讀取 iris 檔案，並進行 sepal_length 和 petal_width 的雙變數探索。
1. 讀取 titanic 檔案，並找出與 Survived 相關性最高的 3 個特徵。
1. 讀取 titanic 檔案，並分析 Embarked 與 Survived 的關係。
1. 讀取 boston 檔案，並進行 RM 和 MEDV 的雙變數探索。
### 6.5.7 練習題解答
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# 1. 讀取 iris 檔案，並進行 sepal_length 和 petal_width 的雙變數探索
iris = sns.load_dataset('iris')
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.scatterplot(x='sepal_length', y='petal_width', data=iris)
print(iris['sepal_length'].corr(iris['petal_width']))

# 2. 讀取 titanic 檔案，並找出與 Survived 相關性最高的 3 個特徵
titanic = sns.load_dataset('titanic')
numeric_titanic = titanic.select_dtypes(include=['number'])
correlation = numeric_titanic.corr()['survived'].abs().sort_values(ascending=False)
print(correlation.head(4))  # survived + 前3重要數值特徵

# 3. 讀取 titanic 檔案，並分析 Embarked 與 Survived 的關係
print(pd.crosstab(titanic['embarked'], titanic['survived']))
plt.subplot(1, 3, 2)
sns.countplot(x='embarked', hue='survived', data=titanic)
print(stats.chi2_contingency(pd.crosstab(titanic['embarked'], titanic['survived'])))

# 4. 讀取 boston.csv 檔案，並進行 RM 和 MEDV 的雙變數探索
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
boston = pd.read_csv(url)
plt.subplot(1, 3, 3)
sns.scatterplot(x='rm', y='medv', data=boston)

# 調整子圖間距（上、下、左、右、橫向間距、縱向間距）
plt.subplots_adjust(
    left=0.1, right=0.9,     # 圖的左右邊界（0 到 1 之間）
    top=0.9, bottom=0.1,     # 上下邊界
    wspace=0.3, hspace=0.4    # 子圖之間的間距（horizontal, vertical）
)

plt.show()
print(boston['rm'].corr(boston['medv']))
```
![image](https://hackmd.io/_uploads/B1_DStUJeg.png)

## 6.6 綜合應用

### 6.6.1 綜合應用簡介

綜合應用是將前面章節學到的描述性統計分析、相關性分析和資料分佈探索等方法，應用於實際資料集，以解決實際問題的過程。它可以幫助我們更深入地了解資料，發現有價值的資訊，並為後續的機器學習建模提供基礎。

### 6.6.2 綜合應用的步驟

1.  **資料載入與初步探索：** 載入資料集，並使用 `head()`、`info()` 和 `describe()` 等函式進行初步探索。
2.  **資料清理與預處理：** 處理缺失值、異常值和重複值，並進行資料轉換和特徵工程。
3.  **描述性統計分析：** 計算均值、中位數、標準差和百分位數，了解資料的集中趨勢和離散程度。
4.  **相關性分析：** 計算皮爾森相關係數，了解變數之間的線性關係。
5.  **資料分佈探索：** 繪製直方圖、箱形圖、核密度估計圖和小提琴圖，了解資料的分佈情況。
6.  **特徵探索：** 進行單變數、雙變數和多變數探索，了解特徵的特性和關係。
7.  **結果總結與報告：** 總結探索結果，並撰寫報告。

### 6.6.3 實例分析：鐵達尼號（Titanic）資料集

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 資料載入與初步探索
url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
titanic = pd.read_csv(url)
print(titanic.head())
print(titanic.info())
print(titanic.describe())

# 2. 資料清理與預處理
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)
titanic.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 3. 描述性統計分析
print(titanic.describe())

# 4. 相關性分析
correlation_matrix = titanic.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# 5. 資料分佈探索
sns.histplot(titanic['Age'])
plt.show()
sns.boxplot(x='Pclass', y='Fare', data=titanic)
plt.show()

# 6. 特徵探索
sns.countplot(x='Sex', hue='Survived', data=titanic)
plt.show()
sns.violinplot(x='Survived', y='Age', data=titanic)
plt.show()

# 7. 結果總結與報告
print("綜合探索結果：...")
```
![image](https://hackmd.io/_uploads/HJ3Q_t8keg.png)

### 6.6.4 綜合應用範例：波士頓房價預測資料集
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 資料載入與初步探索
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
boston = pd.read_csv(url)
print(boston.head())
print(boston.info())
print(boston.describe())

# 2. 資料清理與預處理
# 波士頓房價資料集通常不需要太多清理，但我們可以檢查缺失值
print(boston.isnull().sum())

# 3. 描述性統計分析
print(boston.describe())

# 4. 相關性分析
correlation_matrix = boston.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# 5. 資料分佈探索
sns.histplot(boston['medv'])
plt.show()
sns.scatterplot(x='rm', y='medv', data=boston)
plt.show()

# 6. 特徵探索
sns.pairplot(boston[['medv', 'rm', 'lstat', 'ptratio']])
plt.show()

# 7. 結果總結與報告
print("綜合探索結果：...")

```
![image](https://hackmd.io/_uploads/rymxYF8yxx.png)
![image](https://hackmd.io/_uploads/Hk-ZtYUJxx.png)
![image](https://hackmd.io/_uploads/r1uZKKUkee.png)


### 6.6.5 綜合應用範例：鳶尾花（Iris）資料集
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 資料載入與初步探索
iris = sns.load_dataset('iris')
print(iris.head())
print(iris.info())
print(iris.describe())

# 2. 資料清理與預處理
# 鳶尾花資料集通常不需要清理

# 3. 描述性統計分析
print(iris.describe())

# 4. 相關性分析
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
correlation_matrix = iris[features].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# 5. 資料分佈探索
sns.pairplot(iris, hue='species')
plt.show()

# 6. 特徵探索
plt.figure(figsize=(12, 8))
i=1
for column in iris.columns[:-1]:
    plt.subplot(2, 2, i)
    sns.violinplot(x='species', y=column, data=iris)
    i+=1
plt.show()

# 7. 結果總結與報告
print("綜合探索結果：...")

```
![image](https://hackmd.io/_uploads/BJQaKY8yle.png)
![image](https://hackmd.io/_uploads/ByKTFYUJgx.png)
![image](https://hackmd.io/_uploads/HJkAtY81xx.png)

### 6.6.6 練習題
1. 使用鐵達尼號資料集，分析 Parch 和 SibSp 對 Survived 的影響。
1. 使用波士頓房價預測資料集，分析 LSTAT 和 MEDV 的關係。
1. 使用鳶尾花資料集，分析 sepal_length、sepal_width、petal_length 和 petal_width 之間的關係。

### 6.6.7 練習題解答
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 使用鐵達尼號資料集，分析 Pclass 和 Siblings/Spouses Aboard 對 Survived 的影響
url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
titanic = pd.read_csv(url)
sns.countplot(x='Pclass', hue='Survived', data=titanic)
plt.show()
sns.countplot(x='Siblings/Spouses Aboard', hue='Survived', data=titanic)
plt.show()

# 2. 使用波士頓房價預測資料集，分析 lstat 和 medv 的關係
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
boston = pd.read_csv(url)
sns.scatterplot(x='lstat', y='medv', data=boston)
plt.show()

# 3. 使用鳶尾花資料集，分析 sepal_length、sepal_width、petal_length 和 petal_width 之間的關係
iris = sns.load_dataset('iris')
sns.pairplot(iris)
plt.show()
```
![image](https://hackmd.io/_uploads/SypzqK8klg.png)
![image](https://hackmd.io/_uploads/S1GQcFIJxe.png)
![image](https://hackmd.io/_uploads/H1xFmctIkge.png)
![image](https://hackmd.io/_uploads/r1Am9KUJgl.png)
