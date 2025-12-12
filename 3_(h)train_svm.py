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

print(f"train_raw shape: {train_raw.shape}")
print(f"test_raw shape: {test_raw.shape}")

print(f"train_raw columns: {train_raw.columns}")
print(f"test_raw columns: {test_raw.columns}")

print(f"train_raw head: {train_raw.head()}")
print(f"test_raw head: {test_raw.head()}")

train_raw.head()
test_raw.head()


