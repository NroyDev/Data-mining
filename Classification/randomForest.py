import numpy as np
import pandas as pd
import decisionTree
from measure import measure

forest = []     # global forest 

# build the randomForest by many decisonTree
def fit(data, labels, treeNum=10, featureNum=5):
    rowLen, colLen = data.shape     # store the row ans col length of data

    # build the decisionTree*treeNum
    for _ in range(treeNum):
        # Bootstrap sample: random choose the row of data and can choose the same one 
        indices = np.random.choice(rowLen, rowLen, True)     
        sub_data = data[indices]
        sub_labels = labels[indices]

        # build the dacisionTree
        selected = np.random.choice(colLen, featureNum, False)      # randomly choose the part of feature
        tree = decisionTree.build_tree(sub_data[:, selected], sub_labels)   # use the partial data to build tree
        forest.append((tree, selected))     # build the forest

# predict the decisionTree
def pred(tree, train):
    # until to find the label
    while tree.label is None:
        if train[tree.feature] <= tree.value:
            tree = tree.child['<=']
        else:
            tree = tree.child['>']
    return tree.label

# predict the randomForest
def randomForest_predict(train):
    predictions = []    # store the every predict in every tree

    # every tree predict the whole data => we have treeNum prediction
    for tree, features in forest:
        for row in train:
            predictions.append(pred(tree, row[features]))   # only choose the sub feature which train the tree

    # every tree voting to the answer
    predictions = np.array(predictions).reshape((len(forest), train.shape[0]))  # reshape to (treeNum, testRowLen)
    vote = np.zeros(train.shape[0])

    for i in range(train.shape[0]):
        vote[i] = max(set(predictions[:, i]), key=predictions[:, i].tolist().count) # choose the label which most freq
        
    return vote

if __name__ == "__main__":
    test_data_A     = pd.read_csv(r'實驗A/test_data.csv')
    train_data_A    = pd.read_csv(r'實驗A/train_data.csv')

    train_datas = train_data_A.iloc[:, :8].values       # shape (N, 8)
    train_ans   = train_data_A.iloc[:, 8].values        # shape (N,)
    test_datas  = test_data_A.iloc[:, :8].values        # shape (N, 8)
    test_ans    = test_data_A.iloc[:, 8].values         # shape (N,)

    fit(train_datas, train_ans)
    predictions = randomForest_predict(test_datas)
    confusion_mtx,accuracy,precision,recall,f1 = measure(predictions,test_ans)
    print("Confusion\n",confusion_mtx)
    print("accuracy\t",accuracy)
    print("precision\t",precision)
    print("recall\t\t",recall)
    print("f1\t\t",f1)