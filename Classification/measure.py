# https://medium.com/@imirene/python%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%88%86%E9%A1%9E%E6%A8%A1%E5%9E%8B%E7%9A%845%E5%80%8B%E8%A9%95%E4%BC%B0%E6%8C%87%E6%A8%99-3260f116ce47
import numpy as np

def measure(predictions,actuals):
    TP = predictions[(predictions == 1) & (predictions==actuals)].size  # 他有你預測有
    TN = predictions[(predictions == 0) & (predictions==actuals)].size  # 他沒有你預測沒有
    FP = predictions[(predictions == 1) & (predictions!=actuals)].size  # 他沒有你預測有 誤會別人
    FN = predictions[(predictions == 0) & (predictions!=actuals)].size  # 他有你預測沒有 誤診
    # print(f"TP = {TP}")
    # print(f"TN = {TN}")
    # print(f"FP = {FP}")
    # print(f"FN = {FN}")

    confusion_mtx = np.array([[TP,FP],[FN,TN]])
    accuracy    = (TP+TN) / (TP+FP+FN+TN)
    precision   = TP / (TP+FP)  # 你預測說有犯罪的人中 有多少%是真的犯罪
    recall      = TP / (TP+FN)  # 所有有病的人中 抓出幾%有病
    f1 = 2 * precision*recall / (precision+recall)

    return confusion_mtx,accuracy,precision,recall,f1

if __name__ == "__main__":
    N = 10
    pred = np.random.choice([0,1],size=10)
    act = np.random.choice([0,1],size=10)
    print("PREDICT:",pred)
    print("ACTUAL :",act)
    measure(pred,act)
