# 實作 Logistic Regression 時用到的參考資料
# https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/learning-model-linear-regression-%E8%88%87logistic-regression%E5%B7%AE%E7%95%B0-5e0d9321261f
# https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5
# https://flag-editors.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%8B%95%E6%89%8B%E5%81%9Alesson-10-%E5%88%B0%E5%BA%95cross-entropy-loss-logistic-loss-log-loss%E6%98%AF%E4%B8%8D%E6%98%AF%E5%90%8C%E6%A8%A3%E7%9A%84%E6%9D%B1%E8%A5%BF-%E4%B8%8A%E7%AF%87-2ebb74d281d
import numpy as np
learnRate = 0.0001
MaxIter = 1000
weight = np.array([])
b = 0
train_std = np.array([])
train_mean= np.array([])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def LogisticRegression_train(train_datas,train_ans):
    train_datas = train_datas.copy()
    train_ans = train_ans.copy()
    global weight,b
    global train_std,train_mean
    # init
    n = train_datas.shape[0]
    dim = train_datas.shape[1]
    weight = np.zeros(dim)
    b = 0
    # 標準化資料 防止因為不同特徵有不同範圍的關係 
    # 造成不同特徵有不同的影響力
    train_std  = np.zeros(dim)
    train_mean = np.zeros(dim)
    for i in np.arange(dim):
        train_std[i]   = train_datas[:,i].std()
        train_mean[i]  = train_datas[:,i].mean()
        train_datas[:,i] = (train_datas[:,i]-train_mean[i])/train_std[i]

    # 對lossfunction梯度下降
    # 找出一組合是的weight
    # 計算lossfunction梯度的數學推倒可以在下面的這兩個連結找到
    # https://medium.com/analytics-vidhya/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d
    # https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression
    for i in np.arange(MaxIter):
        z = (train_datas @ weight) + b
        fz = sigmoid(z)
        dw = (1/n) * (train_datas.T @ (fz - train_ans))
        db = (1/n) * np.sum(fz - train_ans)
        weight  -= learnRate * dw
        b       -= learnRate * db


def LogisticRegression_predict(test_datas):
    # 標準化資料 防止因為不同特徵有不同範圍的關係 
    # 造成不同特徵有不同的影響力
    for i in np.arange(test_datas.shape[1]):
        # std   = test_datas[:,i].std()
        # mean  = test_datas[:,i].mean()
        test_datas[:,i] = (test_datas[:,i]-train_mean[i])/train_std[i]

    z = (test_datas @ weight.T) + b
    predictions = np.round(sigmoid(z))
    return predictions
    


if __name__ == "__main__":
    import pandas as pd
    from measure import measure
    test_data_A = pd.read_csv(r'實驗B/test_data.csv')
    train_data_A = pd.read_csv(r'實驗B/train_data.csv')

    train_datas = train_data_A.iloc[:, :8].values         # shape (N, 8)
    train_ans   = train_data_A.iloc[:, 8].values     # shape (N,)
    LogisticRegression_train(train_datas,train_ans)
    print("weight: ",weight)
    print("bias: ",b)

    test_datas = test_data_A.iloc[:, :8].values         # shape (N, 8)
    test_ans   = test_data_A.iloc[:, 8].values     # shape (N,)
    predictions = LogisticRegression_predict(test_datas)

    confusion_mtx,accuracy,precision,recall,f1 = measure(predictions,test_ans)
    print("Confusion\n",confusion_mtx)
    print("accuracy\t",accuracy)
    print("precision\t",precision)
    print("recall\t\t",recall)
    print("f1\t\t",f1)