import pandas as pd
import numpy as np

def Kmeans_predict(K,data,max_iter):
    idx = np.random.choice(data.shape[0], size=K, replace=False)
    # center = np.random.randn(K,data.shape[1])
    center = data[idx]
    label = np.full(data.shape[0],-1)
    for t in np.arange(max_iter):
        dirty = False
        for i in np.arange(data.shape[0]):
            current = data[i]
            dis = np.sum((center-current)**2,axis=1)
            newlabel = np.argmin(dis)
            if(newlabel!=label[i]):
                dirty = True
                label[i] = newlabel
        for i in np.arange(K):
            center[i] = np.sum(data[label==i],axis=0)/np.sum(label==i)
        if(not dirty):
            break
    return label

# # 包含確定資料
# def Kmeans2_predict(K1,K2,known_pred,data,unknown_idx,max_iter):
#     known_pred -= 1

#     unknown_data = data[unknown_idx]
#     known_data = np.delete(data, unknown_idx,axis=0)
#     known_ans = np.delete(known_pred, unknown_idx,axis=0)

#     idx = np.random.choice(unknown_data.shape[0], size=K2, replace=False)
#     unknonw_center = data[idx]

#     known_center = []
#     for i in np.arange(K1):
#         if(np.sum(known_ans==i)>0):
#             known_center.append(np.sum(known_data[known_ans==i],axis=0)/(np.sum(known_ans==i)))
#         else:
#             known_center.append(np.random.randn(data.shape[1]))

#     center = np.concatenate([known_center,unknonw_center],axis=0)

#     label = known_pred.copy()
#     for t in np.arange(max_iter):
#         for i in unknown_idx:
#             current = data[i]
#             dis = np.sum((center-current)**2,axis=1)
#             label[i] = np.argmin(dis)
#         for i in np.arange(K1+K2):
#             if(np.sum(label==i)>0):
#                 center[i] = np.sum(data[label==i],axis=0)/np.sum(label==i)

#     label+=1

#     return label
