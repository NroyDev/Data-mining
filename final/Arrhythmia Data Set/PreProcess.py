import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import TomekLinks

# 僅load資料和fillna
def load1(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    
    train_data = train_data.values
    test_data = test_data.values

    return train_data,train_ans,test_data,test_ans

# load資料和fillna 標準化
def load2(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data,train_ans,test_data,test_ans


# load資料和fillna 標準化 刪除全部都一樣的column
def load3(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # 把全部值都一樣的col drop掉
    mask = train_data.nunique() > 1
    train_data = train_data.loc[:, mask]
    test_data = test_data.loc[:, mask]
    # 一樣的row drop掉
    mask = train_data.T.nunique() > 1
    train_data = train_data.loc[mask]
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data,train_ans,test_data,test_ans


# load資料和fillna 標準化 刪除全部都一樣的column
def load3(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # 把全部值都一樣的col drop掉
    mask = train_data.nunique() > 1
    train_data = train_data.loc[:, mask]
    test_data = test_data.loc[:, mask]
    # 一樣的row drop掉
    mask = train_data.T.nunique() > 1
    train_data = train_data.loc[mask]
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data,train_ans,test_data,test_ans


# load資料和fillna 標準化 undersampling
def load4(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    # undersampling
    train_data, train_ans = TomekLinks().fit_resample(train_data, train_ans)

    return train_data,train_ans,test_data,test_ans

# load資料和fillna 標準化 刪除全部都一樣的column undersampling
def load5(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # 把全部值都一樣的col drop掉
    mask = train_data.nunique() > 1
    train_data = train_data.loc[:, mask]
    test_data = test_data.loc[:, mask]
    # 一樣的row drop掉
    mask = train_data.T.nunique() > 1
    train_data = train_data.loc[mask]
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    # undersampling
    train_data, train_ans = TomekLinks().fit_resample(train_data, train_ans)

    return train_data,train_ans,test_data,test_ans


# load資料和fillna 標準化
def load6(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    # outlier # SVC似乎有變好一點點點點 但速度方面變快蠻多的 / KNN用了會過度自信 / RandomForest Logistic Regression用了會降低一點ACC
    for i in range(train_data.shape[1]):
        mask = np.abs(train_data[:,i])<=3
        train_data = train_data[mask]
        train_ans = train_ans[mask]

    return train_data,train_ans,test_data,test_ans


# load資料和fillna 標準化 刪除全部都一樣的column undersampling
def load7(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # 把全部值都一樣的col drop掉
    mask = train_data.nunique() > 1
    train_data = train_data.loc[:, mask]
    test_data = test_data.loc[:, mask]
    # 一樣的row drop掉
    mask = train_data.T.nunique() > 1
    train_data = train_data.loc[mask]
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    # outlier # SVC似乎有變好一點點點點 但速度方面變快蠻多的 / KNN用了會過度自信 / RandomForest Logistic Regression用了會降低一點ACC
    for i in range(train_data.shape[1]):
        mask = np.abs(train_data[:,i])<=3
        train_data = train_data[mask]
        train_ans = train_ans[mask]
    # undersampling
    train_data, train_ans = TomekLinks().fit_resample(train_data, train_ans)
    

    return train_data,train_ans,test_data,test_ans


# 篩選相關係數較高的特徵出來 => 變糟了
def load8(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    train = np.concatenate([train_data.values,train_ans.reshape(train_ans.size,1)],axis=1)
    train = pd.DataFrame(train)
    targetCorr = abs(train.corr().values[:,-1])
    mask = (targetCorr>0.1)[:-1]
    # print(mask)
    # print(mask.shape)
    train_data = train_data.loc[:,mask]
    test_data = test_data.loc[:,mask]

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())

    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data,train_ans,test_data,test_ans

# load資料和fillna 標準化 PCA
def load9(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(train_data,train_ans)
    train_data = pca.transform(train_data)
    test_data  = pca.transform(test_data)

    return train_data,train_ans,test_data,test_ans

# load資料和fillna 標準化 刪除全部都一樣的column outlier
def load10(train_data_path,train_ans_path,test_data_path,test_ans_path):
    # 載入資料
    train_data = pd.read_csv(train_data_path,header=None,dtype=float)
    train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

    test_data = pd.read_csv(test_data_path,header=None,dtype=float)
    test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

    # 填充NA
    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(train_data[col].mean())
    # 把全部值都一樣的col drop掉
    mask = train_data.nunique() > 1
    train_data = train_data.loc[:, mask]
    test_data = test_data.loc[:, mask]
    # 一樣的row drop掉
    mask = train_data.T.nunique() > 1
    train_data = train_data.loc[mask]
    # sklearn 要求 label 從 0 開始，所以轉換為 0~7
    train_ans -= 1
    # 標準化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    # outlier # SVC似乎有變好一點點點點 但速度方面變快蠻多的 / KNN用了會過度自信 / RandomForest Logistic Regression用了會降低一點ACC
    for i in range(train_data.shape[1]):
        mask = np.abs(train_data[:,i])<=3
        train_data = train_data[mask]
        train_ans = train_ans[mask]
    

    return train_data,train_ans,test_data,test_ans

# def pre(train_data_path,train_ans_path,test_data_path,test_ans_path):
#     # 載入資料
#     train_data = pd.read_csv(train_data_path,header=None,dtype=float)
#     train_data = train_data.drop(train_data.columns[13], axis=1)
#     train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

#     test_data = pd.read_csv(test_data_path,header=None,dtype=float)
#     test_data = test_data.drop(test_data.columns[13], axis=1)
#     test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

#     # 填充NA
#     for col in train_data.columns:
#         train_data[col] = train_data[col].fillna(train_data[col].mean())
#         test_data[col] = test_data[col].fillna(train_data[col].mean())
#     # 把全部值都一樣的col drop掉
#     mask = train_data.nunique() > 1
#     train_data = train_data.loc[:, mask]
#     test_data = test_data.loc[:, mask]
#     # sklearn 要求 label 從 0 開始，所以轉換為 0~7
#     train_ans -= 1
#     # 標準化
#     scaler = StandardScaler()
#     train_data = scaler.fit_transform(train_data)
#     test_data = scaler.transform(test_data)

#     return train_data,train_ans,test_data,test_ans

# def pre2(train_data_path,train_ans_path,test_data_path,test_ans_path):
#     # 載入資料
#     train_data = pd.read_csv(train_data_path,header=None,dtype=float)
#     train_data = train_data.drop(train_data.columns[13], axis=1)
#     train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

#     test_data = pd.read_csv(test_data_path,header=None,dtype=float)
#     test_data = test_data.drop(test_data.columns[13], axis=1)
#     test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

#     # 填充NA
#     for col in train_data.columns:
#         train_data[col] = train_data[col].fillna(train_data[col].mean())
#         test_data[col] = test_data[col].fillna(train_data[col].mean())
#     # 把全部值都一樣的col drop掉
#     mask = train_data.nunique() > 1
#     train_data = train_data.loc[:, mask]
#     test_data = test_data.loc[:, mask]
#     # 一樣的row drop掉
#     mask = train_data.T.nunique() > 1
#     train_data = train_data.loc[mask]
#     test_data = test_data.loc[mask]

#     # sklearn 要求 label 從 0 開始，所以轉換為 0~7
#     train_ans -= 1
#     # 標準化
#     scaler = StandardScaler()
#     train_data = scaler.fit_transform(train_data)
#     test_data = scaler.transform(test_data)

#     # undersampling
#     train_data, train_ans = TomekLinks().fit_resample(train_data, train_ans)

#     # oversampling SMOTE # 使用 SMOTE oversampling 後變糟了
#     # temp = train_data[train_ans==7]
#     # train_data = np.delete(train_data, train_ans==7,axis=0)
#     # train_ans = np.delete(train_ans, train_ans==7,axis=0)
#     # train_data, train_ans = SMOTE(k_neighbors = 2).fit_resample(train_data, train_ans)
#     # train_data = np.concatenate([train_data,temp],axis=0)
#     # train_ans = np.concatenate([train_ans,[7]],axis=0)

#     return train_data,train_ans,test_data,test_ans


# def pre3(train_data_path,train_ans_path,test_data_path,test_ans_path):
#     # 載入資料
#     train_data = pd.read_csv(train_data_path,header=None,dtype=float)
#     train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

#     test_data = pd.read_csv(test_data_path,header=None,dtype=float)
#     test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

#     # 刪掉太多NA的 col
#     mask = train_data.isna().sum(axis=0)/train_data.shape[1] < 0.5
#     test_data = test_data.loc[:,mask]
#     train_data = train_data.loc[:,mask]
    
#     # train_data[13] = train_data[13].fillna(0)
#     # test_data[13] = test_data[13].fillna(0)
#     # 填充NA
#     for col in train_data.columns:
#         train_data[col] = train_data[col].fillna(train_data[col].mean())
#         test_data[col] = test_data[col].fillna(train_data[col].mean())
#     # 把全部值都一樣的col drop掉
#     mask = train_data.nunique(axis=0) > 1
#     train_data = train_data.loc[:, mask]
#     test_data = test_data.loc[:, mask]
#     # 一樣的row drop掉
#     mask = train_data.nunique(axis=1) > 1
#     train_data = train_data.loc[mask]
#     test_data = test_data.loc[mask]

#     # sklearn 要求 label 從 0 開始，所以轉換為 0~7
#     train_ans -= 1
#     # 標準化
#     scaler = StandardScaler()
#     train_data = scaler.fit_transform(train_data)
#     test_data = scaler.transform(test_data)

    
#     # outlier # SVC似乎有變好一點點點點 但速度方面變快蠻多的 / KNN用了會過度自信 / RandomForest Logistic Regression用了會降低一點ACC
#     # for i in range(train_data.shape[1]):
#     #     mask = np.abs(train_data[:,i])<=3
#     #     train_data = train_data[mask]
#     #     train_ans = train_ans[mask]
#     #     # if(np.sum(~mask)>0):
#     #     #     print("OUTLIER",np.sum(~mask))

#     # undersampling
#     train_data, train_ans = TomekLinks().fit_resample(train_data, train_ans)

#     return train_data,train_ans,test_data,test_ans


# # 篩選相關係數較高的特徵出來 => 變糟了
# def pre4(train_data_path,train_ans_path,test_data_path,test_ans_path):
#     # 載入資料
#     train_data = pd.read_csv(train_data_path,header=None,dtype=float)
#     train_ans = pd.read_csv(train_ans_path,header=None,dtype=int).values.ravel()

#     test_data = pd.read_csv(test_data_path,header=None,dtype=float)
#     test_ans = pd.read_csv(test_ans_path,header=None,dtype=int).values.ravel()

#     train = np.concatenate([train_data.values,train_ans.reshape(train_ans.size,1)],axis=1)
#     train = pd.DataFrame(train)
#     targetCorr = abs(train.corr().values[:,-1])
#     mask = (targetCorr>0.1)[:-1]
#     # print(mask)
#     # print(mask.shape)
#     train_data = train_data.loc[:,mask]
#     test_data = test_data.loc[:,mask]

#     # 刪掉太多NA的 col
#     mask = train_data.isna().sum(axis=0)/train_data.shape[1] < 0.5
#     test_data = test_data.loc[:,mask]
#     train_data = train_data.loc[:,mask]
#     # 填充NA
#     for col in train_data.columns:
#         train_data[col] = train_data[col].fillna(train_data[col].mean())
#         test_data[col] = test_data[col].fillna(train_data[col].mean())
#     # 把全部值都一樣的col drop掉
#     mask = train_data.nunique(axis=0) > 1
#     train_data = train_data.loc[:, mask]
#     test_data = test_data.loc[:, mask]
#     # 一樣的row drop掉
#     mask = train_data.nunique(axis=1) > 1
#     train_data = train_data.loc[mask]
#     test_data = test_data.loc[mask]

#     # sklearn 要求 label 從 0 開始，所以轉換為 0~7
#     train_ans -= 1
#     # 標準化
#     scaler = StandardScaler()
#     train_data = scaler.fit_transform(train_data)
#     test_data = scaler.transform(test_data)

#     # undersampling
#     train_data, train_ans = TomekLinks().fit_resample(train_data, train_ans)

#     return train_data,train_ans,test_data,test_ans


    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # X = pca.fit_transform(Xsys)