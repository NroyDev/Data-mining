from Classification import Cla
# from PreProcess import pre,pre2,pre3,pre4
import numpy as np
from sklearn.cluster import KMeans
from Kmeans import Kmeans_predict

def my_kmeans_ver(train_data,train_ans,test_data,test_ans):
    # 分類
    acc_avg = 0
    miss = 0
    for i in range(N):
        preds, unknown_idx = Cla(train_data,train_ans,test_data,test_ans)
        
        if(OUTPUT):
            preds_temp = np.delete(preds, unknown_idx)
            ans_temp = np.delete(test_ans, unknown_idx)
            print("Classifier ACC (排除不確定之資料): ",np.sum(preds_temp==ans_temp)/len(ans_temp))
            print("Classifier ACC (加入不確定之資料): ",np.sum(preds==test_ans)/len(test_ans))

        # 分群
        if(unknown_idx.size>5):
            unknown_data = test_data[unknown_idx]
            labels = Kmeans_predict(5,unknown_data,1000)

            values, counts = np.unique(labels, return_counts=True)
            order = values[counts.argsort()[::-1]]
            used = []
            for i in order:
                values, counts = np.unique(test_ans[unknown_idx][labels==i], return_counts=True)
                maxidx = 0
                for j in range(values.size):
                    if(values[maxidx]<=8 or (values[maxidx] in used) or counts[maxidx]<counts[j]):
                        maxidx = j
                # maxidx = np.argmax(counts)
                labels[labels==i] = values[maxidx]
                used.append(values[maxidx])
            preds[unknown_idx] = labels

            # 最終結果
            if(OUTPUT):
                print("最後(Classifier再Clustur後) ACC: ",np.sum(preds==test_ans)/len(test_ans))
            acc_avg += np.sum(preds==test_ans)/len(test_ans)
        else:
            if(OUTPUT):
                print("ERROR 沒有不確定的")
            miss+=1
        if(OUTPUT):
            print("--------------------------------------------------------")
            print("預測\n",preds)
            print("答案\n",test_ans)

    if(N-miss==0):
        print(np.sum(preds==test_ans)/len(test_ans),"\n備註: Classifier過度自信 他沒有不確定的")
        return np.sum(preds==test_ans)/len(test_ans)
    else:
        acc_avg/=(N-miss)
        print("avgACC =",acc_avg)
        return acc_avg

def sklearn_kmeans_ver(train_data,train_ans,test_data,test_ans):
    # 分類
    acc_avg = 0
    miss = 0
    for i in range(N):
        preds, unknown_idx = Cla(train_data,train_ans,test_data,test_ans)
        
        if(OUTPUT):
            preds_temp = np.delete(preds, unknown_idx)
            ans_temp = np.delete(test_ans, unknown_idx)
            print("Classifier ACC (排除不確定之資料): ",np.sum(preds_temp==ans_temp)/len(ans_temp))
            print("Classifier ACC (加入不確定之資料): ",np.sum(preds==test_ans)/len(test_ans))

        # 分群
        if(unknown_idx.size>5):
            unknown_data = test_data[unknown_idx]
            kmeans = KMeans(n_clusters=5,max_iter=1000)
            kmeans.fit(unknown_data)
            labels = kmeans.labels_.copy()

            values, counts = np.unique(labels, return_counts=True)
            order = values[counts.argsort()[::-1]]
            used = []
            for i in order:
                values, counts = np.unique(test_ans[unknown_idx][labels==i], return_counts=True)
                maxidx = 0
                for j in range(values.size):
                    if(values[maxidx]<=8 or (values[maxidx] in used) or counts[maxidx]<counts[j]):
                        maxidx = j
                # maxidx = np.argmax(counts)
                labels[labels==i] = values[maxidx]
                used.append(values[maxidx])
            preds[unknown_idx] = labels

            # 最終結果
            if(OUTPUT):
                print("最後(Classifier再Clustur後) ACC: ",np.sum(preds==test_ans)/len(test_ans))
            acc_avg += np.sum(preds==test_ans)/len(test_ans)
        else:
            if(OUTPUT):
                print("ERROR 沒有不確定的")
            miss+=1
        if(OUTPUT):
            print("--------------------------------------------------------")
            print("預測\n",preds)
            print("答案\n",test_ans)

    if(N-miss==0):
        print(np.sum(preds==test_ans)/len(test_ans),"\n備註: Classifier過度自信 他沒有不確定的")
        return np.sum(preds==test_ans)/len(test_ans)
    else:
        acc_avg/=(N-miss)
        print("avgACC =",acc_avg)
        return acc_avg

import time
from PreProcess import load1,load2,load3,load4,load5,load6,load7,load8,load9,load10
def run(ver):

    st = time.time()
    train_data,train_ans,test_data,test_ans = load1("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("幾乎沒有任何處理: \n\t",end="")
    acc1 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load2("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化: \n\t",end="")
    acc2 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對幾乎沒有任何處理進步 {(acc2-acc1)*100:.2f}%")

    print("-----------------------------------------------------------------------------")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load3("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + 刪除全部都一樣的column: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load4("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + undersampling: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load6("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + 刪除outlier: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load9("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + PCA: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load8("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + 挑出相關係數較高的特徵去訓練: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")
    
    print("-----------------------------------------------------------------------------")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load5("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + 刪除全部都一樣的column + undersampling: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")

    st = time.time()
    train_data,train_ans,test_data,test_ans = load10("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + 刪除全部都一樣的column + 刪除outlier: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")
    
    st = time.time()
    train_data,train_ans,test_data,test_ans = load7("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + 刪除全部都一樣的column + undersampling + 刪除outlier: \n\t",end="")
    acc3 = ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    print(f"\t相對標準化進步 {(acc3-acc2)*100:.2f}%")
    
OUTPUT = False
N = 100

if __name__== '__main__':
    run(my_kmeans_ver)

    print("---------------------------------------------------------")
    OUTPUT = True
    N = 1
    st = time.time()
    train_data,train_ans,test_data,test_ans = load7("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
    print("標準化 + 刪除全部都一樣的column + undersampling + 刪除outlier: \n",end="")
    acc3 = my_kmeans_ver(train_data,train_ans,test_data,test_ans)
    et = time.time()
    print(f"\t耗時= {et-st:.2f} 秒")
    