import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

print("Import Complete")

def Cla(train_data,train_ans,test_data,test_ans):
    # clf = LogisticRegression(max_iter=1000)
    # clf = RandomForestClassifier()
    # clf = KNeighborsClassifier()
    clf = SVC(probability=True)
    clf.fit(train_data, train_ans)

    # 預測並計算不確定度
    probs = clf.predict_proba(test_data)
    preds = clf.predict(test_data) + 1
    max_probs = np.max(probs, axis=1)
    uncertainty = 1 - max_probs

    # 不確定
    threshold = 0.6
    unknown_idx = np.where(max_probs < threshold)[0]
    # print(f"不確定數量:{len(unknown_idx)}")
    # print("不確定idx:",unknown_idx)

    # preds_sure = np.delete(preds, unknown_idx)
    # ans_sure = np.delete(test_ans, unknown_idx)
    # data_notsure = test_data[unknown_idx,...]
    # preds_notsure = preds[unknown_idx]
    # ans_notsure = test_ans[unknown_idx]

    return preds, unknown_idx