#!/usr/bin/env python3
import csv
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneOut


def wdbc_normal():
    with open('./wdbc.csv') as f:
        normal_data = []
        wdbc_data = []
        for row in csv.reader(f):
            if row[1] == 'M':
                wdbc_data.append(
                    [float(datum) for datum in row[2:]]
                )
            else:
                normal_data.append(
                    [float(datum) for datum in row[2:]]
                )
    label = [str(i) for i in range(1, len(row[2:])+1)]
    return normal_data, wdbc_data, label


# 読み込み
random.seed(1)
normal_data, wdbc_data, label = wdbc_normal()
normal_data = random.sample(normal_data, len(normal_data))
wdbc_data = random.sample(wdbc_data, len(wdbc_data))
all_data = normal_data + wdbc_data
all_label = ["Normal"] * len(normal_data) + ["WDBC"] * len(wdbc_data)

# parameters = {
#     'n_estimators': [2, 3, 4, 5, 10, 20, 30],
#     'max_features': [3, 6, 9],
#     'random_state': [0],
#     'min_samples_split': [2, 3, 4, 5, 10, 15],
#     'max_depth': [3, 4, 5, 10],
#     "criterion": ["gini"]
# }
# clf = GridSearchCV(
#     RandomForestClassifier(),
#     parameters,
#     cv=3,
#     n_jobs=-1
# )
# clf.fit(all_data, all_label)
# print(clf.best_estimator_)

# L法
print("--- L法 ---")
loo = LeaveOneOut()
count = [[0, 0], [0, 0]]
for train_index, test_index in loo.split(all_data):
    train_data = [all_data[i] for i in train_index]
    train_label = [all_label[i] for i in train_index]
    test_data = [all_data[i] for i in test_index]
    test_label = [all_label[i] for i in test_index]
    clf = RandomForestClassifier(
        bootstrap=True, ccp_alpha=0.0, class_weight=None,
        criterion='gini', max_depth=10, max_features=6,
        max_leaf_nodes=None, max_samples=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=3,
        min_weight_fraction_leaf=0.0, n_estimators=30,
        n_jobs=None, oob_score=False, random_state=0, verbose=0,
        warm_start=False)
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)
    if predicted == test_label and test_label == ["Normal"]:
        count[0][0] += 1
    elif predicted != test_label and test_label == ["Normal"]:
        count[1][0] += 1
    elif predicted == test_label and test_label == ["WDBC"]:
        count[1][1] += 1
    elif predicted != test_label and test_label == ["WDBC"]:
        count[0][1] += 1

l_normal = count[0][0]/len(normal_data)
l_wdbc = count[1][1]/len(wdbc_data)
l_total = (count[0][0]+count[1][1])/len(all_data)

print("--- cm ---")
print("pred\\act| Normal | WDBC")
print("Normal   |  {0:4}  | {1}".format(count[0][0], count[0][1]))
print("WDBC     |  {0:4}  | {1}".format(count[1][0], count[1][1]))
print("         | {0:>.4f} | {1:>.4f} | {2:>.4f}".format(
    l_normal, l_wdbc, l_total))
print("L法(Normal)   :%f" % (l_normal))
print("L法(WDBC)     :%f" % (l_wdbc))
print("L法(Total)    :%f" % (l_total))

# R法
print("--- R法 ---")
clf = RandomForestClassifier(
    bootstrap=True, ccp_alpha=0.0, class_weight=None,
    criterion='gini', max_depth=10, max_features=6,
    max_leaf_nodes=None, max_samples=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=3,
    min_weight_fraction_leaf=0.0, n_estimators=30,
    n_jobs=None, oob_score=False, random_state=0, verbose=0,
    warm_start=False)
clf.fit(all_data, all_label)
predicted = clf.predict(all_data)
cm = confusion_matrix(all_label, predicted)

l_normal = (cm[0][0])/len(normal_data)
l_wdbc = (cm[1][1])/len(wdbc_data)
l_total = (cm[0][0]+cm[1][1])/len(all_data)

print("--- cm ---")
print("pred\\act| Normal | WDBC")
print("Normal   |  {0:4}  | {1}".format(cm[0][0], cm[1][0]))
print("WDBC     |  {0:4}  | {1}".format(cm[0][1], cm[1][1]))
print("         | {0:>.4f} | {1:>.4f} | {2:>.4f}".format(
    l_normal, l_wdbc, l_total))
print("R法(Normal)   :%f" % (l_normal))
print("R法(WDBC)     :%f" % (l_wdbc))
print("R法(Total)    :%f" % (l_total))

fti = clf.feature_importances_
print('Feature Importances:')
for i, feat in enumerate(label):
    print('\t{0:25s} : {1:>.6f}'.format(feat, fti[i]))
