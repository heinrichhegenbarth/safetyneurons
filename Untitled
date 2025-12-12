# %%
# Imports

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # try with and without
from sklearn.metrics import accuracy_score

# %%
# Data

BASE_PATH = "/home/hheg_stli/AML_Project/data/classifier_data/"

train_raw = pd.read_csv(f'{BASE_PATH}/training_data.csv') 
test_raw = pd.read_csv(f'{BASE_PATH}/testing_data.csv')
safety_neurons = pd.read_csv(f'{BASE_PATH}/top_safetyneurons.csv')

print(f"train_raw shape: {train_raw.shape}") # train_raw shape: (3539, 92161)
print(f"test_raw shape: {test_raw.shape}") # test_raw shape: (885, 92161)

print(f"train_raw columns: {train_raw.columns}") # train_raw columns: Index(['0', '0.1', '1', '2', '3', '4', '5', '6', '7', '8', ... '92150', '92151', '92152', '92153', '92154', '92155', '92156', '92157', '92158', '92159'], dtype='object', length=92161)
print(f"test_raw columns: {test_raw.columns}") # test_raw columns: Index(['0', '0.1', '1', '2', '3', '4', '5', '6', '7', '8', ... '92150', '92151', '92152', '92153', '92154', '92155', '92156', '92157', '92158', '92159'], dtype='object', length=92161)

print(f"train_raw head: {train_raw.head()}")
# train_raw head:
#    0       0.1         1         2  ...     92156     92157     92158     92159
# 0  0  1.737604 -0.727799  0.592120  ...  5.859375 -1.464414 -1.692391  4.476562
# 1  1  2.069557 -0.768563  0.550159  ...  7.764648 -1.887205 -4.629663  5.392672
# 2  0  2.399745 -0.862502  0.679774  ...  7.459930 -0.451220 -2.626685  5.868959
# 3  1  1.948037 -0.711568  0.529242  ...  7.766727 -1.136900 -0.970902  4.374890
# 4  0  2.405242 -0.948589  0.724948  ...  8.237904 -0.933513 -2.494405  4.847888
# [5 rows x 92161 columns]

print(f"test_raw head: {test_raw.head()}")
# test_raw head:    
#    0       0.1         1         2  ...     92156     92157     92158     92159
# 0  0  1.646549 -0.664287  0.481407  ...  7.078157 -1.189141 -0.145154  2.335179
# 1  0  1.646227 -0.648662  0.550845  ...  7.949013 -1.198925 -2.087807  4.292069
# 2  1  1.329642 -0.646561  0.447622  ...  7.811215 -1.132533  0.427059  4.451159
# 3  1  1.941946 -0.696636  0.599327  ...  5.779220 -1.472692 -0.804944  3.709466
# 4  1  1.993527 -0.855597  0.631592  ...  6.235236 -0.066358 -0.464318  2.917654
# [5 rows x 92161 columns]
