import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from KNN_sklearn import KNN_sklearn
from KNN import KNN_vectorized

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def KNN_sklearn_test(train_data, test_data, max_k):
    accuracies = []
    accs = {}
    for k in range(1, max_k, 2):
        pred = KNN_sklearn(train_data, test_data, k)
        accuracy = accuracy_score(test_data.iloc[:, 8], pred)
        print(f"{'sklearn KNN':^10s} : {accuracy:12.8f}")
        accuracies.append(accuracy)
        accs[k] = accuracy
    best_k, best_acc = max(accs.items(), key=lambda item: item[1])
    return best_k, best_acc, accuracies

def KNN_test(train_data, test_data, max_k):
    accuracies = []
    accs = {}
    for k in range(1, max_k, 2):
        pred = KNN_vectorized(train_data, test_data, k)
        accuracy = accuracy_score(test_data.iloc[:, 8], pred)
        print(f"{'KNN':^10s} : {accuracy:12.8f}")
        accuracies.append(accuracy)
        accs[k] = accuracy
    best_k, best_acc = max(accs.items(), key=lambda item: item[1])
    return best_k, best_acc, accuracies

def test_others(skObj,train_data, test_data):
    X_train = train_data.iloc[:, :8]
    y_train = train_data.iloc[:, 8]
    X_test  = test_data.iloc[:, :8]
    y_test  = test_data.iloc[:, 8]
    clf     = skObj
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc

if __name__ == "__main__":
    # 載入資料
    train_data  = pd.read_csv('實驗A/train_data.csv')
    test_data   = pd.read_csv('實驗A/test_data.csv')

    # 測試 KNN
    MaxN = 50
    sklearn_k, sklearn_acc, sk_knn_accs = KNN_sklearn_test(train_data, test_data, MaxN)
    KNN_k, KNN_acc, knn_accs            = KNN_test(train_data, test_data, MaxN)

    # 測試其他演算法
    logreg_acc  = test_others(LogisticRegression(max_iter=1000)             ,train_data, test_data)
    svm_acc     = test_others(SVC()                                         ,train_data, test_data)
    dt_acc      = test_others(DecisionTreeClassifier(random_state=42)       ,train_data, test_data)
    rf_acc      = test_others(RandomForestClassifier(random_state=42)       ,train_data, test_data)
    gb_acc      = test_others(GradientBoostingClassifier(random_state=42)   ,train_data, test_data)
    nb_acc      = test_others(GaussianNB()                                  ,train_data, test_data)
    mlp_acc     = test_others(MLPClassifier(random_state=42, max_iter=1000) ,train_data, test_data)
    lda_acc     = test_others(LinearDiscriminantAnalysis()                  ,train_data, test_data)
    qda_acc     = test_others(QuadraticDiscriminantAnalysis()               ,train_data, test_data)

    # 輸出總結結果
    print("\nSummary:")
    print(f"{'Algorithm':<20s}{'k':>5s}{'Accuracy':>12s}")
    print(f"{'sklearn KNN':<20s}{sklearn_k:>5d}{sklearn_acc:>12.4f}")
    print(f"{'KNN (vectorized)':<20s}{KNN_k:>5d}{KNN_acc:>12.4f}")
    print(f"{'LogisticRegression':<20s}{'':>5s}{logreg_acc:>12.4f}")
    print(f"{'SVM':<20s}{'':>5s}{svm_acc:>12.4f}")
    print(f"{'DecisionTree':<20s}{'':>5s}{dt_acc:>12.4f}")
    print(f"{'RandomForest':<20s}{'':>5s}{rf_acc:>12.4f}")
    print(f"{'GradBoost':<20s}{'':>5s}{gb_acc:>12.4f}")
    print(f"{'NaiveBayes':<20s}{'':>5s}{nb_acc:>12.4f}")
    print(f"{'MLP':<20s}{'':>5s}{mlp_acc:>12.4f}")
    print(f"{'LDA':<20s}{'':>5s}{lda_acc:>12.4f}")
    print(f"{'QDA':<20s}{'':>5s}{qda_acc:>12.4f}")

    algo_names  = np.array([f'sklearn KNN, K={sklearn_k}',f'KNN (vectorized), K={KNN_k}','LogisticRegression','SVM','DecisionTree','RandomForest','GradBoost','NaiveBayes','MLP','LDA','QDA'])
    algo_accs   = np.array([sklearn_acc,KNN_acc,logreg_acc,svm_acc,dt_acc,rf_acc,gb_acc,nb_acc,mlp_acc,lda_acc,qda_acc])
    algo_colors = np.array(["k","k","grey","grey","grey","grey","grey","grey","grey","grey","grey","grey"])
    sort_idx    = np.argsort(algo_accs)
    algo_names  = algo_names[sort_idx]
    algo_accs   = algo_accs[sort_idx]
    algo_colors = algo_colors[sort_idx]

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(8,6))
    fig.tight_layout()
    gd = gridspec.GridSpec(2, 2)
    ax0 = plt.subplot(gd[0,0])
    ax1 = plt.subplot(gd[0,1])
    ax2 = plt.subplot(gd[1,1])
    ax0.grid()
    ax1.grid()
    ax2.grid(axis='x')

    ax0.plot(np.arange(1,MaxN,2),sk_knn_accs)
    ax1.plot(np.arange(1,MaxN,2),knn_accs)
    ax2.barh(np.arange(algo_names.size),algo_accs,color=algo_colors,tick_label=algo_names,height=0.5)
    ax1.set(title="KNN implement by us")
    ax0.set(title="KNN build in sklearn")
    ax2.set(title="The Accuracies of different algorithms",axisbelow=True,xlim=[0.6,0.85])
    plt.show()

    # ax2.barh(np.arange(algo_names.size),algo_accs,color=algo_colors,tick_label=algo_names,height=0.25)