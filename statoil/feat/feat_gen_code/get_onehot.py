import pandas as pd
import numpy as np

train = pd.read_json("../../input/train.json")
test = pd.read_json("../../input/test.json")
train['inc_angle'] = train['inc_angle'].replace('na', 0)
test['inc_angle'] = test['inc_angle'].replace('na', 0)

train_oh = pd.DataFrame()
test_oh = pd.DataFrame()

train_oh['id'] = train['id']
test_oh['id'] = test['id']
train_oh['inc_angle'] = train['inc_angle']
test_oh['inc_angle'] = test['inc_angle']

raw_vals = np.unique(train['inc_angle'])
print(raw_vals)
print(raw_vals.shape)
val_map = {}
for i in range(len(raw_vals)):
    val_map[raw_vals[i]] = i
print(val_map)
train_oh['ang'] = train['inc_angle'].map(val_map)
test_oh['ang'] = test['inc_angle'].map(val_map)

print(train_oh.head())
print(test_oh.head())
