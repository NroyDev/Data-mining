from measure import measure
from GaussianNavieBayes import GaussianNavieBayes_predict,GaussianNavieBayes_Train
from LogisticRegression import LogisticRegression_predict,LogisticRegression_train
from KNN import KNN_vectorized as KNN
from decisionTree import decisionTree_train,decisionTree_predict
from randomForest import fit as randomForest_train
from randomForest import randomForest_predict

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test(train_func,test_func,train_csv, test_csv):
    train_datas = train_csv.iloc[:, :8].values
    train_ans   = train_csv.iloc[:, 8].values
    test_datas  = test_csv.iloc[:, :8].values
    test_ans    = test_csv.iloc[:, 8].values
    
    train_func(train_datas, train_ans)
    predictions = test_func(test_datas)
    confusion_mtx,accuracy,precision,recall,f1 =  measure(predictions,test_ans)
    return accuracy,precision,recall,f1

def test_KNN(train_set, test_set, k):
    predictions = KNN(train_set,test_set,k)
    test_ans    = test_csv.iloc[:, 8].values
    confusion_mtx,accuracy,precision,recall,f1 = measure(predictions,test_ans)
    return accuracy,precision,recall,f1

if __name__ == '__main__':
    path = "實驗A/"
    train_csv  = pd.read_csv(path+'train_data.csv')
    test_csv   = pd.read_csv(path+'test_data.csv')
    print("LOAD COMPLETE")

    algos = ["SVC","Decision Tree","Random Forest","Gradient Boosting","MLP","Linear Discriminant Analysis","Quadratic Discriminant Analysis","Gaussian NavieBayes","Logistic Regression","KNN"]
    algos = [l.replace(" ","\n") for l in algos]
    algos = np.array(algos)

    print("START PREDICT")
    temp = []
    clf = SVC()
    temp.append(test(clf.fit,clf.predict,train_csv,test_csv))
    print("SVC COMPLETE")
    # clf = DecisionTreeClassifier(random_state=42)
    # temp.append(test(clf.fit,clf.predict,train_csv,test_csv))
    # clf = RandomForestClassifier(random_state=42)
    # temp.append(test(clf.fit,clf.predict,train_csv,test_csv))
    temp.append(test(decisionTree_train,decisionTree_predict,train_csv,test_csv))
    print("decisionTree COMPLETE")
    temp.append(test(randomForest_train,randomForest_predict,train_csv,test_csv))
    print("randomForest COMPLETE")
    clf = GradientBoostingClassifier(random_state=42)
    temp.append(test(clf.fit,clf.predict,train_csv,test_csv))
    print("Gradient Boosting COMPLETE")
    clf = MLPClassifier(random_state=42, max_iter=1000)
    temp.append(test(clf.fit,clf.predict,train_csv,test_csv))
    print("MLP COMPLETE")
    clf = LinearDiscriminantAnalysis()
    temp.append(test(clf.fit,clf.predict,train_csv,test_csv))
    print("Linear Discriminant COMPLETE")
    clf = QuadraticDiscriminantAnalysis()
    temp.append(test(clf.fit,clf.predict,train_csv,test_csv))
    print("Quadratic Discriminant COMPLETE")
    temp.append(test(GaussianNavieBayes_Train,GaussianNavieBayes_predict,train_csv,test_csv))
    print("Gaussian Navie Bayes COMPLETE")
    temp.append(test(LogisticRegression_train,LogisticRegression_predict,train_csv,test_csv))
    print("Logistic Regression COMPLETE")
    temp.append(test_KNN(train_csv,test_csv,17))
    print("KNN COMPLETE")
    print("END PREDICT")


    m = np.array(temp)
    for i,j in zip(m,algos):
        print(j,i)
    
    print("START PLOT RESULT")
    plt.rc('axes', axisbelow=True)
    measure_name = ["accuracy","precision","recall","f1"]
    for i in np.arange(4):
        sort_idx = (m[:,i]).argsort()
        m = m[sort_idx]
        algos = algos[sort_idx]
        fig = plt.figure(figsize=(6,8))
        plt.barh(np.arange(algos.size),m[:,i],tick_label=algos,height=0.5)
        plt.title(f"{measure_name[i]} in Different Alogrithms")
        plt.xlabel(measure_name[i])
        plt.ylabel("Algoritms")
        # plt.xlim([0.5,1])
        plt.xlim(left = 0.5)
        plt.tight_layout()
        plt.grid(axis = 'x',linestyle='dashed')
        fig.savefig(f"./result/{measure_name[i]}.png")


    temp = []
    MAXK = 51
    for k in np.arange(3,MAXK,2):
        temp.append(test_KNN(train_csv,test_csv,k))
    temp = np.array(temp)

    fig = plt.figure(figsize=(6,4))
    fig.tight_layout()
    for i in np.arange(4):
        plt.plot(np.arange(3,MAXK,2),temp[:,i],label=measure_name[i])
    plt.legend()
    plt.title("KNN's accuracy/precision/recall/f1 in different K")
    plt.xlabel("K")
    plt.grid(linestyle='dashed')
    fig.savefig(f"./result/KNN.png")
    print("END PLOT RESULT")
    print("EOP")


