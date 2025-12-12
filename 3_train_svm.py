# %%
# Imports

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import joblib
from sklearn.decomposition import PCA

BASE_PATH = "./data/classifier_data"
OUTPUT_PATH = "./models/classifiers"
# %%
# Data
train_raw = pd.read_csv(f"{BASE_PATH}/training_data.csv")
test_raw = pd.read_csv(f"{BASE_PATH}/testing_data.csv")
safety_neurons = pd.read_csv(f"{BASE_PATH}/top_safetyneurons.csv")

# limit data for testing to 10 samples
train_raw = train_raw.sample(10)
test_raw = test_raw.sample(10)

print(f"train_raw shape: {train_raw.shape}")  # train_raw shape: (3539, 92161)
print(f"test_raw shape: {test_raw.shape}")  # test_raw shape: (885, 92161)
print(
    f"safety_neurons shape: {safety_neurons.shape}"
)  # safety_neurons shape: (4608, 1)

print(
    f"train_raw columns: {train_raw.columns}"
)  # train_raw columns: Index(['0', '0.1', '1', '2', '3', '4', '5', '6', '7', '8', ... '92150', '92151', '92152', '92153', '92154', '92155', '92156', '92157', '92158', '92159'], dtype='object', length=92161)
print(
    f"test_raw columns: {test_raw.columns}"
)  # test_raw columns: Index(['0', '0.1', '1', '2', '3', '4', '5', '6', '7', '8', ... '92150', '92151', '92152', '92153', '92154', '92155', '92156', '92157', '92158', '92159'], dtype='object', length=92161)
print(
    f"safety_neurons columns: {safety_neurons.columns}"
)  # safety_neurons columns: Index(['0'], dtype='object', length=1)

print(f"train_raw head: {train_raw.head(2)}")
# train_raw head:
#    0       0.1         1         2  ...     92156     92157     92158     92159
# 0  0  1.737604 -0.727799  0.592120  ...  5.859375 -1.464414 -1.692391  4.476562
# 1  1  2.069557 -0.768563  0.550159  ...  7.764648 -1.887205 -4.629663  5.392672
# [2 rows x 92161 columns]

print(f"test_raw head: {test_raw.head(2)}")
# test_raw head:
#    0       0.1         1         2  ...     92156     92157     92158     92159
# 0  0  1.646549 -0.664287  0.481407  ...  7.078157 -1.189141 -0.145154  2.335179
# 1  0  1.646227 -0.648662  0.550845  ...  7.949013 -1.198925 -2.087807  4.292069
# [2 rows x 92161 columns]

# %%
# preparing the data

# filter by safety neurons
train_temp = train_raw.iloc[:, 0]
test_temp = test_raw.iloc[:, 0]


safety_set = set(safety_neurons.loc[:, 'neuron_index'])
column_mask = train_temp.columns.isin(safety_set)

X_train_sn = train_temp.loc[:, column_mask]
y_train_sn = train_raw.iloc[:, 0]
X_test_sn = test_raw.loc[:, column_mask]
y_test_sn = test_raw.iloc[:, 0]

print(f"X_train_sn shape: {X_train_sn.shape}")
print(f"y_train_sn shape: {y_train_sn.shape}")
print(f"X_test_sn shape: {X_test_sn.shape}")
print(f"y_test_sn shape: {y_test_sn.shape}")

# pca
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(train_raw.iloc[:, 1:])
X_test_pca = pca.transform(test_raw.iloc[:, 1:])
y_train_pca = train_raw.iloc[:, 0]
y_test_pca = test_raw.iloc[:, 0]

print(f"number principal components: {pca.n_components_}")

# full data
X_train_full = train_raw.iloc[:, 1:]
X_test_full = test_raw.iloc[:, 1:]
y_train_full = train_raw.iloc[:, 0]
y_test_full = test_raw.iloc[:, 0]

# %%
# training the models

# svm with safety neurons
model_sn = SVC()
model_sn.fit(X_train_sn, y_train_sn)
joblib.dump(model_sn, f"{OUTPUT_PATH}/model_sn.pkl")

# svm with pca
model_pca = SVC()
model_pca.fit(X_train_pca, y_train_pca)
joblib.dump(model_pca, f"{OUTPUT_PATH}/model_pca.pkl")

# svm with full data
model_full = SVC()
model_full.fit(X_train_full, y_train_full)
joblib.dump(model_full, f"{OUTPUT_PATH}/model_full.pkl")
# %%
# predictions

predictions_train_sn = model_sn.predict(X_train_sn)
predictions_test_sn = model_sn.predict(X_test_sn)

predictions_train_pca = model_pca.predict(X_train_pca)
predictions_test_pca = model_pca.predict(X_test_pca)

predictions_train_full = model_full.predict(X_train_full)
predictions_test_full = model_full.predict(X_test_full)


# svm with safety neurons
def evaluate_model(model_name, y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    cm = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions)
    print(f"----------------{model_name}----------------")
    print(f"accuracy: {accuracy}")
    print(f"recall: {recall}")
    print(f"confusion matrix: {cm}")
    print(f"classification report: {report} \n \n")


evaluate_model("safety neurons", y_test_sn, predictions_test_sn)
evaluate_model("pca", y_test_pca, predictions_test_pca)
evaluate_model("full", y_test_full, predictions_test_full)
