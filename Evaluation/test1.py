#
import pandas as pd
import time

path_train = "data/dm/train.csv"  # 训练文件
data = pd.read_csv(path_train, nrows=5)
l = []
time_list = [0, 0, 0, 0]
time_res = []
time_ = data['TIME'].tolist()
for timeStamp in time_:
    timeArray = time.localtime(timeStamp)
    # 获取某时
    l.append(time.strftime("%Y-%m-%d %H:%M:%S", timeArray)[11:13])
    # l=[5,4]
    time_list = [0, 0, 0, 0]

    index1 = int((int(l[0]) - 4) / 6)
    index2 = int((int(l[-1]) - 4) / 6)
    # index1 = int((int(4)-4)/6)
    # index2 = int((int(10)-4)/6)
    index1 = 0 if (index1 >= 3 or index1 < 0 or int(l[0]) < 4) else index1 + 1
    index2 = 0 if (index2 >= 3 or index2 < 0 or int(l[-1]) < 4) else index2 + 1

    if index1 <= index2:
        for i in range(index1, index2 + 1):
            time_list[i] = 1
    else:
        for i in range(index1, 4):
            time_list[i] = 1
        for i in range(0, index2 + 1):
            time_list[i] = 1
    time_res.append(time_list)
