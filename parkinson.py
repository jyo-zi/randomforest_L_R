#!/usr/bin/env python3
import csv
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneOut


def parkinson_normal():
    with open('./parkinsons.csv') as f:
        normal_data = []
        parkinson_data = []
        for row in csv.DictReader(f):
            if row["status"] == '0':
                del row["name"], row["status"]
                normal_data.append([float(datum) for datum in row.values()])
            else:
                del row["name"], row["status"]
                parkinson_data.append([float(datum) for datum in row.values()])
            label = [str(key) for key in row.keys()]
    return normal_data, parkinson_data, label


# 読み込み
random.seed(1)
normal_data, parkinson_data, label = parkinson_normal()
normal_data = random.sample(normal_data, len(normal_data))
parkinson_data = random.sample(parkinson_data, len(parkinson_data))
all_data = normal_data + parkinson_data
all_label = ["Normal"] * len(normal_data) + ["Parkinson"] * len(parkinson_data)

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
        bootstrap=True, ccp_alpha=0.0,
        class_weight=None, criterion='gini',
        max_depth=10, max_features=6,
        max_leaf_nodes=None, max_samples=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=3,
        min_weight_fraction_leaf=0.0, n_estimators=10,
        n_jobs=None, oob_score=False, random_state=0,
        verbose=0, warm_start=False)
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)
    if predicted == test_label and test_label == ["Normal"]:
        count[0][0] += 1
    elif predicted != test_label and test_label == ["Normal"]:
        count[1][0] += 1
    elif predicted == test_label and test_label == ["Parkinson"]:
        count[1][1] += 1
    elif predicted != test_label and test_label == ["Parkinson"]:
        count[0][1] += 1

l_normal = count[0][0]/len(normal_data)
l_parkinson = count[1][1]/len(parkinson_data)
l_total = (count[0][0]+count[1][1])/len(all_data)

print("--- cm ---")
print("pred\\act| Normal | Parkinson")
print("Normal   |  {0:4}  | {1}".format(count[0][0], count[0][1]))
print("Parkinson|  {0:4}  | {1}".format(count[1][0], count[1][1]))
print("         | {0:>.4f} | {1:>.4f} | {2:>.4f}".format(
    l_normal, l_parkinson, l_total))
print("L法(Normal)   :%f" % (l_normal))
print("L法(Parkinson):%f" % (l_parkinson))
print("L法(Total)    :%f" % (l_total))

# R法
print("--- R法 ---")
clf = RandomForestClassifier(
    bootstrap=True, ccp_alpha=0.0,
    class_weight=None, criterion='gini',
    max_depth=10, max_features=6,
    max_leaf_nodes=None, max_samples=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=3,
    min_weight_fraction_leaf=0.0, n_estimators=10,
    n_jobs=None, oob_score=False, random_state=0,
    verbose=0, warm_start=False)
clf.fit(all_data, all_label)
predicted = clf.predict(all_data)
cm = confusion_matrix(all_label, predicted)

l_normal = (cm[0][0])/len(normal_data)
l_parkinson = (cm[1][1])/len(parkinson_data)
l_total = (cm[0][0]+cm[1][1])/len(all_data)

print("--- cm ---")
print("pred\\act| Normal | Parkinson")
print("Normal   |  {0:4}  | {1}".format(cm[0][0], cm[1][0]))
print("Parkinson|  {0:4}  | {1}".format(cm[0][1], cm[1][1]))
print("         | {0:>.4f} | {1:>.4f} | {2:>.4f}".format(
    l_normal, l_parkinson, l_total))
print("R法(Normal)   :%f" % (l_normal))
print("R法(Parkinson):%f" % (l_parkinson))
print("R法(Total)    :%f" % (l_total))

fti = clf.feature_importances_
print('Feature Importances:')
for i, feat in enumerate(label):
    print('\t{0:25s} : {1:>.6f}'.format(feat, fti[i]))
