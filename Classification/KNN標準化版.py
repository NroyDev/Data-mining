import pandas as pd
import numpy as np
from collections import Counter

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x-y) ** 2))

def KNN(train_set, test_set, k):
    train_datas = train_set.iloc[:, :8].values          # shape (N, 8)
    train_ans   = train_set.iloc[:, 8].values           # shape (N,)
    test_datas  = test_set.iloc[:, :8].values           # shape (N, 8)

    # 標準化資料 防止因為不同特徵有不同範圍的關係 
    # 造成不同特徵有不同的影響力
    for i in np.arange(train_datas.shape[1]):
        train_std   = train_datas[:,i].std()
        train_mean  = train_datas[:,i].mean()
        train_datas[:,i]    = (train_datas[:,i]-train_mean)/train_std
        test_datas[:,i]     = (test_datas[:,i]-train_mean)/train_std

    predictions = []
    for test_point in test_datas:
        # step 1. 計算與每個點的 Ecuild distance
        distances = [euclidean_distance(test_point, train_point) for train_point in train_datas]
        # step 2. 找出最近的K個點
        k_indices = np.argsort(distances)[:k]
        k_labels = [train_ans[i] for i in k_indices]
        # step 3. 這些點投票(看哪種多)去預測
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

def KNN_vectorized(train_set, test_set, k):
    # 取出特徵與標籤，假設前8個欄位為特徵，第9欄為標籤
    train_features = train_set.iloc[:, :8].values  # shape (N, 8)
    train_labels   = train_set.iloc[:, 8].values     # shape (N,)
    test_features  = test_set.iloc[:, :8].values    # shape (M, 8)
    
    # 標準化資料 防止因為不同特徵有不同範圍的關係 
    # 造成不同特徵有不同的影響力
    for i in np.arange(train_features.shape[1]):
        train_std   = train_features[:,i].std()
        train_mean  = train_features[:,i].mean()
        train_features[:,i] = (train_features[:,i]-train_mean)/train_std
        test_features[:,i] = (test_features[:,i]-train_mean)/train_std
    
    # 向量化計算歐式距離：
    # 利用公式：||a-b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    dists = np.sqrt( np.maximum(0,
        np.sum(test_features**2, axis=1)[:, None] +
        np.sum(train_features**2, axis=1)[None, :] -
        2 * np.dot(test_features, train_features.T)
    ))
    
    predictions = []
    # 對每個測試樣本
    for i in range(dists.shape[0]):
        # 使用 np.argpartition 找出距離最小的 k 個索引
        k_indices = np.argpartition(dists[i], k)[:k]
        k_labels = train_labels[k_indices]
        # 多數投票決定預測標籤
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    
    return np.array(predictions)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_data_A = pd.read_csv(r'實驗A/test_data.csv')
    train_data_A = pd.read_csv(r'實驗A/train_data.csv')

    Accuracies = []
    print("K/Accuracy of KNN with K")
    for k in np.arange(1, 50, 2):
        count = 0
        size = test_data_A.shape[0]
        predictions = KNN_vectorized(train_data_A, test_data_A, k)
        for i in np.arange(size):
            if test_data_A.iloc[i, 8] ==  predictions[i]:
                count+=1
        Accuracies.append(count/size)
        print(k, Accuracies[-1])

    plt.plot(np.arange(1,50,2),Accuracies)
    plt.grid()
    plt.show()