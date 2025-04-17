import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def KNN_sklearn(train_data, test_data, k):
    X_train = train_data.iloc[:, :8]  # 前8欄作為特徵
    y_train = train_data.iloc[:, 8]   # 第9欄為標籤
    X_test = test_data.iloc[:, :8]

    knn = KNeighborsClassifier(n_neighbors=k)

    # 訓練模型
    knn.fit(X_train, y_train)

    # 使用模型對測試集進行預測
    predictions = knn.predict(X_test)

    return predictions

if __name__ == "__main__":
    # 載入訓練集與測試集
    train_data = pd.read_csv('實驗A/train_data.csv')
    test_data = pd.read_csv('實驗A/test_data.csv')

    # 分離特徵與標籤
    X_train = train_data.iloc[:, :8]  # 前8欄作為特徵
    y_train = train_data.iloc[:, 8]   # 第9欄為標籤
    X_test = test_data.iloc[:, :8]
    y_test = test_data.iloc[:, 8]


    for k in range(3, 50, 2):
        # 建立 KNN 模型，設定 k 值（例如 k=3）
        # k = 3
        knn = KNeighborsClassifier(n_neighbors=k)

        # 訓練模型
        knn.fit(X_train, y_train)

        # 使用模型對測試集進行預測
        predictions = knn.predict(X_test)

        # 計算準確率
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy =", accuracy)
