import numpy as np
from normal2data import normal2data

# 1) Linear‐axis test
norm_lin = np.array([2.0, 4.0, 6.0])
axis_meta_lin = {'S': [2.0, 2.0], 'type': 0}
data_lin = normal2data(norm_lin, axis_meta_lin, is_y=False)
print("Linear axis: norm=", norm_lin, "=> data=", data_lin)
# expected: [0, 1, 2]

# 2) Log‐axis test
norm_log = np.array([2.0, 3.0, 4.0])
axis_meta_log = {'S': [1.0, 1.0], 'type': 1}
data_log = normal2data(norm_log, axis_meta_log, is_y=True)
print("Log axis:    norm=", norm_log, "=> data=", data_log)
# expected: [10, 100, 1000]
