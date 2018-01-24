import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
from mpl_toolkits.axes_grid1 import ImageGrid
import random
random.seed(1)

train = pd.read_json("../../input/train.json")
agg_df = train.groupby('inc_angle').agg({"is_iceberg": [len, np.sum]}).sort_values([('is_iceberg', 'len')], ascending=False)
print(agg_df[0:30])
