# **tradercompany method**

trader-company methodを実装しました。  
より論文実装に近い形で実装していますが、大枠は以下のqiita記事を参考にしました。

## 参考文献
- https://arxiv.org/abs/2012.10215
- https://qiita.com/yotapoon/items/1214218c7459ad69db3e


## 使い方

1. ライブラリのインポート

```
import os
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tradercompany
from tradercompany.activation_funcs import identity, ReLU, sign, tanh
from tradercompany.binary_operators import add, diff, get_x, get_y, multiple, x_is_greater_than_y
from tradercompany.trader import Trader
from tradercompany.company import Company

%matplotlib inline


SEED = 2021
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_all_seeds(SEED)
```

2. データ準備

pandas.DataFrame形式の時系列``df_y``を準備します。訓練用のデータと検証用に分割します。

```
T_train = 800
df_y_train = df_y.iloc[:T_train, :]
df_y_test = df_y.iloc[T_train:, :]
```

3. trader-company methodのパラメータを指定する

```
activation_funcs = [identity, ReLU, sign, tanh]
binary_operators = [max, min, add, diff, multiple, get_x, get_y, x_is_greater_than_y]
stock_names = ["stock0", "stock1"]
time_window = 200
delay_time_max = 2
num_factors_max = 4
```

4. モデルを構築する

```
model = Company(stock_names, 
                num_factors_max, 
                delay_time_max, 
                activation_funcs, 
                binary_operators, 
                num_traders=40, 
                Q=0.2, 
                time_window=time_window, 
                how_recruit="random")
```

5. 学習する

```
model.fit(df_y_train)
```

6. モデルの保存

```
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

7. 次の時刻の予測

```
# 時刻t+1の予測
model.aggregate()
```

8-1. 検証用データに対する予測(tuningなし)

```
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

errors_test_notuning = []
for i, row in df_y_test.iterrows():
    prediction_test = model.aggregate()
    errors_test_notuning.append(np.abs(row.values - prediction_test))
    
    # tuning==Falseの場合、データが追加されても重みの更新などパラメータは変わらない
    model.fit_new_data(row.to_dict(), tuning=False)
```
8-2. 検証用データに対する予測(tuningあり)

```
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

errors_test_tuning = []
for i, row in df_y_test.iterrows():
    prediction_test = model.aggregate()
    errors_test_tuning.append(np.abs(row.values - prediction_test))
    
    # tuning==Trueの場合、データが追加された際に重みの更新などパラメータが調整される
    model.fit_new_data(row.to_dict(), tuning=True)
```






