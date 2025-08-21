from PreProcess import load7
from Classification import Cla
from Kmeans import Kmeans_predict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run(train_data,train_ans,test_data,test_ans):
    preds, unknown_idx = Cla(train_data,train_ans,test_data,test_ans)
    preds_temp = np.delete(preds, unknown_idx)
    ans_temp = np.delete(test_ans, unknown_idx)
    # 分群
    if(unknown_idx.size>5):
        unknown_data = test_data[unknown_idx]
        labels = Kmeans_predict(5,unknown_data,1000)
        print(labels)

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

    else:
        print("ERROR")
        return run(train_data,train_ans,test_data,test_ans)

    print("--------------------------------------------------------")
    print("ACC =",np.sum(preds==test_ans)/preds.size)
    print("預測\n",preds)
    print("答案\n",test_ans)
    return preds



train_data,train_ans,test_data,test_ans = load7("train_data.csv","train_label.csv","test_data.csv","test_label.csv")
print("標準化 + 刪除全部都一樣的column + undersampling + 刪除outlier: \n",end="")

def plot_t(train_data,train_ans,test_data,test_ans):
    preds = run(train_data,train_ans,test_data,test_ans)
    # 劃出結果
    from sklearn.manifold import TSNE,MDS
    model = TSNE(n_components=2)
    printData = model.fit_transform(test_data)

    fig = plt.figure(figsize=(8,4))
    ax = fig.subplots(1,2)
    # 正確的
    ax[0].scatter(printData[...,0],printData[...,1],s=10,c=test_ans,cmap="Set3")
    ax[0].set_title("actual ans")
    # 預測的
    ax[1].scatter(printData[...,0],printData[...,1],s=10,alpha=0,c=test_ans,cmap="Set3")
    ax[1].scatter(printData[...,0],printData[...,1],s=10,c=preds,cmap="Set3")
    ax[1].set_title("prediction")
    plt.show()

    # confuion matrix
    from sklearn.metrics import confusion_matrix
    sns.heatmap(confusion_matrix(test_ans,preds),annot=True, cmap='Blues')
    plt.show()

plot_t(train_data,train_ans,test_data,test_ans)