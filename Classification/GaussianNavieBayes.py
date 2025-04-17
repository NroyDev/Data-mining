# 實作 Gaussian Naive Bayes 時用到的參考資料
# https://www.youtube.com/watch?v=H3EjCKtlVog
# https://roger010620.medium.com/%E8%B2%9D%E6%B0%8F%E5%88%86%E9%A1%9E%E5%99%A8-naive-bayes-classifier-%E5%90%ABpython%E5%AF%A6%E4%BD%9C-66701688db02
import numpy as np
typep   = np.array([])
stds    = np.array([[]])
means   = np.array([[]])
TYPES = 2

def GaussianNavieBayes_Train(train_datas,train_ans):
    global stds,means,typep
    # init
    n       = train_datas.shape[0]
    dim     = train_datas.shape[1]
    stds    = np.zeros((TYPES,dim))
    means   = np.zeros((TYPES,dim))
    typep   = np.zeros(TYPES)

    # 因為 GaussianNavieBayes 是用 Gaussian Distribution
    # 作為連續資料的機率的PDF 所以在這裡我們要來計算
    # 在答案是 0 和 1 情況下，不同特徵 Gaussian Distribution 的平均以及標準差
    # 這裡也要順便計算 答案是0和答案是1的機率分別是多少
    for i in np.arange(TYPES):
        for j in np.arange(dim):
            stds[i,j]   = train_datas[train_ans==i,j].std()
            means[i,j]  = train_datas[train_ans==i,j].mean()
        typep[i] = (train_ans[train_ans==i].size)/n

# 取過log的Gaussian Distribution PDF
# 主要是為了防止Underflow
def GaussDistrFunc(type,dim,x):
    std     = stds[type,dim]
    mean    = means[type,dim]
    return -1/2*np.log(2*np.pi*(std**2)) - ((x-mean)**2)/(2*(std**2))

def GaussianNavieBayes_predict(test_datas):
    # init
    n   = test_datas.shape[0]
    dim = test_datas.shape[1]
    predictions = np.zeros(n)

    cnt = 0
    for data in test_datas:
        # 計算在這樣的資料下，是0的機率是多少 是1的機率又是多少
        # 這邊套過log用加的
        ps = np.log(typep).copy()
        for i in np.arange(TYPES):
            for j in np.arange(dim):
                ps[i] += GaussDistrFunc(i,j,data[j])
        # 一樣找最大機率的
        predict = (-ps).argsort()[0]
        predictions[cnt] = predict
        cnt+=1
    
    return predictions


if __name__ == "__main__":
    import pandas as pd
    from measure import measure
    test_data_A     = pd.read_csv(r'實驗A/test_data.csv')
    train_data_A    = pd.read_csv(r'實驗A/train_data.csv')

    train_datas = train_data_A.iloc[:, :8].values         # shape (N, 8)
    train_ans   = train_data_A.iloc[:, 8].values     # shape (N,)
    GaussianNavieBayes_Train(train_datas,train_ans)
    test_datas  = test_data_A.iloc[:, :8].values         # shape (N, 8)
    test_ans    = test_data_A.iloc[:, 8].values     # shape (N,)
    predictions = GaussianNavieBayes_predict(test_datas)

    confusion_mtx,accuracy,precision,recall,f1 = measure(predictions,test_ans)
    print("Confusion\n",confusion_mtx)
    print("accuracy\t",accuracy)
    print("precision\t",precision)
    print("recall\t\t",recall)
    print("f1\t\t",f1)