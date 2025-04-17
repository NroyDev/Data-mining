# https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda
# https://zh.wikipedia.org/zh-tw/%E5%86%B3%E7%AD%96%E6%A0%91
# because sklearn use Gini and binary tree to handle continuous data, so I do the same
import pandas as pd
import numpy as np
from measure import measure

# computing Gini impurity
def Gini(labels):
    length = len(labels)    # total length of labels
    label_freq = {}         # the dict to store the count of every labels

    # counting the frequency
    for label in labels:
        if label not in label_freq.keys():
            label_freq[label] = 0
        label_freq[label] += 1

    # counting the Gini = 1-∑pᵢ²
    gini = 1.0
    for freq in label_freq.values():
        gini -= (freq/length)**2
    
    return gini

# given feature, find the best threshold to cut the data
def best_gain(data, index, labels):
    length = data.shape[1]              # row's length of the data
    feature_col = data[:, index]        # the col of feature in data
    feature_val = set(feature_col)      # all types in this feature

    # if the feature only has one type => do not need to split
    if len(feature_val) == 1:
        return float('inf'), None

    sorted_feature = sorted(feature_val)    # sort the feature to get the thresholds
    thresholds = []                         # store the every threshold((type1+type2)/2, (type2+typ3)/2, ....)
    for i in range(len(sorted_feature)-1):
        thresholds.append((sorted_feature[i]+sorted_feature[i+1])/2)

    best_gini = float('inf')        # store the best Gini value
    best_threshold = None           # store the best threshold value
    
    for threshold in thresholds:
        # turn the labels to two parts: (1) <= threshold (2) > threshold
        left_labels = labels[feature_col <= threshold]
        right_labels = labels[feature_col > threshold]

        # weighted_gini = (nL/N)*Gini(L) + (nR/N)*Gini(R)
        weighted_gini = (len(left_labels)/length)*Gini(left_labels) + (len(right_labels)/length)*Gini(right_labels)

        # to find the minimun weighted_gini => purer split
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_threshold = threshold

    return best_gini, best_threshold

# find the best feature to split the data
def best_split(data, labels):
    length = data.shape[1]      # col's length of the data
    best_feature = -1           # store the index of the best feature
    best_threshold = None       # store the num of the best threshold
    min_gini = float('inf')     # the minimun gini = the best gini

    # to find the best split feature and threshold
    for i in range(length):
        cur_gini, cur_threshold = best_gain(data, i, labels)
        if cur_gini < min_gini:
            best_feature = i
            best_threshold = cur_threshold
            min_gini = cur_gini
    
    return best_feature, best_threshold

# decision tree's node
class Node:
    def __init__(self, feature=None, value=None, label=None):
        self.feature = feature  
        self.value = value      # store the threshold => means it is nonleaf node
        self.label = label
        self.child = {}

def decisionTree_train(train_datas, train_ans):
    global root
    root = build_tree(train_datas, train_ans)

# build the decision tree
def build_tree(data, labels):
    # if the labels are the same => do not need to split => label = label[0]
    if len(labels) == 1:    
        return Node(label = labels[0])
    
    # if no feature we can split => choose the labels which freq is most highest
    if len(data[0]) == 0:   
        return Node(label=max(set(labels), key=labels.tolist().count))
    
    best_feature, best_threshold = best_split(data, labels)     # choose best feature, threshold

    # if no feature or no threshold can help to split => choose the labels which freq is most highest
    if best_feature == -1 or best_threshold == None:    
        return Node(label=max(set(labels), key=labels.tolist().count))
   
    root = Node(feature = best_feature, value = best_threshold)     # create none-leaf node(feature, threshold)

    # turn the data to two parts: (1) <= threshold (2) > threshold
    left_data = data[data[:, best_feature] <= best_threshold]
    right_data = data[data[:, best_feature] > best_threshold]

    # turn the labels to two parts: (1) <= threshold (2) > threshold
    left_labels = labels[data[:, best_feature] <= best_threshold]
    right_labels = labels[data[:, best_feature] > best_threshold]

    # create two child: (1) <= threshold (2) > threshold
    root.child['<='] = build_tree(left_data, left_labels)
    root.child['>'] = build_tree(right_data, right_labels)

    return root

'''
Since other algorithms don't use recursion for prediction, 
a global root is defined to unify the interface (so that only the data needs to be passed in).
'''
root = Node()   # global root

# main func call this to predict test data
def decisionTree_predict(test):
    predictions = []    # store the predict ans by test

    for row in test:
        predictions.append(predict(root, row)) 

    return np.array(predictions)

# the truely predict function
def predict(root, test):
    # find label => isLeaf node
    if root.label is not None:
        return root.label
    
    # recursion to find the label
    feature_value = test[root.feature]                        
    if feature_value <= root.value:                          
        return predict(root.child['<='], test)    
    else:                                                       
        return predict(root.child['>'], test)

if __name__ == "__main__":
    test_data_A     = pd.read_csv(r'實驗A/test_data.csv')
    train_data_A    = pd.read_csv(r'實驗A/train_data.csv')

    train_datas = train_data_A.iloc[:, :8].values       # shape (N, 8)
    train_ans   = train_data_A.iloc[:, 8].values        # shape (N,)
    root = build_tree(train_datas, train_ans)
    # print(root)
    test_datas  = test_data_A.iloc[:, :8].values        # shape (N, 8)
    test_ans    = test_data_A.iloc[:, 8].values         # shape (N,)
    predictions = decisionTree_predict(test_datas)
    confusion_mtx,accuracy,precision,recall,f1 = measure(predictions,test_ans)
    print("Confusion\n",confusion_mtx)
    print("accuracy\t",accuracy)
    print("precision\t",precision)
    print("recall\t\t",recall)
    print("f1\t\t",f1)