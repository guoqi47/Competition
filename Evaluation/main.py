# -*- coding:utf8 -*-
import os
import csv
# import pandas as pd
from sklearn.cluster import KMeans
import gc
from math import radians, cos, sin, asin, sqrt
import pandas
import numpy
import time
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA

t1 = time.time()
gc.collect()


def time_list(times):
    time1 = time.strftime("%H", time.localtime(times[0]))[0:2]
    time2 = time.strftime("%H", time.localtime(times[1]))[0:2]
    time_list = [0, 0, 0, 0]

    index1 = int((int(time1) - 4) / 6)
    index2 = int((int(time2) - 4) / 6)
    index1 = 0 if (index1 >= 3 or index1 < 0 or int(time1) < 4) else index1 + 1
    index2 = 0 if (index2 >= 3 or index2 < 0 or int(time2) < 4) else index2 + 1

    if index1 <= index2:
        for i in range(index1, index2 + 1):
            time_list[i] = 1
    else:
        for i in range(index1, 4):
            time_list[i] = 1
        for i in range(0, index2 + 1):
            time_list[i] = 1
    return time_list


def haversine(zuo):
    lon1, lat1, lon2, lat2 = map(radians, zuo)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


def compress(trip, user_id, trip_id):
    le = len(trip) - 1
    var_v = trip.SPEED.var()
    var_dir = trip.DIRECTION.var()
    ave_h = trip.HEIGHT.mean()
    time_ = trip.TIME.tolist()
    l = []
    for timeStamp in time_:
        timeArray = time.localtime(timeStamp)
        # 获取某时
        l.append(time.strftime("%H", timeArray))

    time_list = [0, 0, 0, 0]
    index1 = int((int(l[0]) - 4) / 6)
    index2 = int((int(l[-1]) - 4) / 6)
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

    dis = haversine(trip.iat[0, 4], trip.iat[0, 3], trip.iat[le, 4], trip.iat[le, 3])
    try:
        ave_v = dis / (trip.iat[le, 7] - trip.iat[0, 7])
    except ZeroDivisionError:
        ave_v = 0

    call = trip.iat[0, 0]
    return pandas.DataFrame(data={'user_id': user_id, 'trip_id': trip_id, 't1': time_list[0], 't2': time_list[1],
                                  't3': time_list[2], 't4': time_list[3],
                                  'var_v': var_v, 'var_dir': var_dir, 'ave_h': ave_h, 'dis': dis, 'Y': trip.iat[0, 9],
                                  'call': call, 'ave_v': ave_v},
                            index=[0])


# def read_csv():
#     """
#     文件读取模块，头文件见columns.
#     :return:
#     """
#     # for filename in os.listdir(path_train):
#     tempdata = pd.read_csv(path_train)[:10000000]
#     tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
#                         "CALLSTATE", "Y"]
#     return tempdata

def proprocess(path):
    if path == path_train:
        csv_chunk = pandas.read_csv(path, iterator=True,
                                    dtype={'TERMINALNO': numpy.int32, 'TIME': numpy.int32,
                                           'TRIP_ID': numpy.int32, 'LONGITUDE': numpy.float64,
                                           'LATITUDE': numpy.float64,
                                           'Y': numpy.float64, 'DIRECTION': numpy.float64,
                                           'HEIGHT': numpy.float64, 'SPEED': numpy.float64,
                                           'CALLSTATE': numpy.int32})
        print('read train set')
        #,nrows=5000000
    else:
        csv_chunk = pandas.read_csv(path, iterator=True,
                                    dtype={'TERMINALNO': numpy.int32, 'TIME': numpy.int32,
                                           'TRIP_ID': numpy.int32, 'LONGITUDE': numpy.float64,
                                           'LATITUDE': numpy.float64,
                                           'Y': numpy.float64, 'DIRECTION': numpy.float64,
                                           'HEIGHT': numpy.float64, 'SPEED': numpy.float64,
                                           'CALLSTATE': numpy.int32})
        print('read test set')

    df = pandas.DataFrame(columns=['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'Y',
                                   'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE'], index=[0])

    curr_id = 1
    try:
        while True:
            df2 = csv_chunk.get_chunk(10000000)

            df = pandas.concat([df, df2])
    except StopIteration:
        pass

    flag = True
    new_row = []
    while flag:
        trip_id = 0
        line_num = 0
        user = df.ix[df['TERMINALNO'] == curr_id]
        if not user.empty:
            user = user.sort_values(by='TIME')
            user = user.reset_index(drop=True)
            user['pre'] = user['TIME'].shift(1)
            user['cha'] = user['TIME'] - user['pre']
            gap = user.ix[user['cha'] >= 1800]

            for index, row in gap.iterrows():
                trip = user.ix[line_num:index - 1]
                le = len(trip) - 1
                line_num = index
                new_row.append([curr_id, trip_id, trip.SPEED, trip.DIRECTION, trip.HEIGHT,
                                [trip.iat[0, 4], trip.iat[0, 3], trip.iat[le, 4], trip.iat[le, 3]],
                                (trip.iat[0, 7], trip.iat[-1, 7]), trip.iat[0, 9], trip.iat[0, 0],
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                trip_id += 1
            trip = user.ix[line_num:]
            le = len(trip) - 1
            new_row.append([curr_id, trip_id, trip.SPEED, trip.DIRECTION, trip.HEIGHT,
                            [trip.iat[0, 4], trip.iat[0, 3], trip.iat[le, 4], trip.iat[le, 3]],
                            (trip.iat[0, 7], trip.iat[-1, 7]), trip.iat[0, 9], trip.iat[0, 0],
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            curr_id += 1
        else:
            break

    user_trips = pandas.DataFrame(new_row, columns=['user_id', 'trip_id', 'speeds', 'dir', 'hig', 'coor',
                                                    'time', 'Y', 'call', "dis", 't1', 't2', 't3', 't4', 'time_list',
                                                    'var_dir', 'ave_v', 'var_v', 'ave_h'])
    user_trips['var_dir'] = user_trips['dir'].map(lambda x: x.var()).map(lambda x: x if pandas.notnull(x) else 0)
    user_trips['var_v'] = user_trips['speeds'].map(lambda x: x.var()).map(lambda x: x if pandas.notnull(x) else 0)
    user_trips['ave_h'] = user_trips['hig'].map(lambda x: x.mean())
    user_trips['dis'] = user_trips['coor'].map(haversine)
    user_trips['ave_v'] = (user_trips['dis'] / user_trips['time'].map(lambda x: x[1] - x[0])).map(
        lambda x: x if pandas.notnull(x) else 0)

    user_trips.pop('speeds')
    user_trips.pop('dir')
    user_trips.pop('hig')
    user_trips.pop('coor')
    user_trips['time_list'] = user_trips['time'].map(time_list)
    user_trips['t1'] = user_trips['time_list'].map(lambda x: x[0])
    user_trips['t2'] = user_trips['time_list'].map(lambda x: x[1])
    user_trips['t3'] = user_trips['time_list'].map(lambda x: x[2])
    user_trips['t4'] = user_trips['time_list'].map(lambda x: x[3])
    user_trips.pop('time_list')
    user_trips.pop('time')
    # user_trips.to_csv('compress.csv')
    del df
    # del user_trips
    gc.collect()
    # process(tempdata)
    # a = pandas.read_csv('compress.csv')
    # print(a[1])
    userIdList = []
    y = []
    userId = user_trips['user_id'][0]
    y.append(user_trips['Y'][0])
    userIdList.append(userId)
    l = []
    lRes = []  # 最后按user_id分的列表
    yRes = [] # 最后按user_id分的列表
    # userIdListRes =

    for index, row in user_trips.iterrows():
        if row['user_id'] == userId:
            l.append(row[3:])

            if index == len(user_trips) - 1:
                lRes.append(l)
                # userIdList.append(int(row['user_id']))
                # y.append(row[2])
        else:
            lRes.append(l)
            userIdList.append(int(row['user_id']))
            y.append(row[2])
            l = []
            l.append(row[3:])
            userId = row['user_id']
    # 进行聚类处理
    if path == path_train:
        clusterCenters = []
        for i in range(len(lRes)):
            if len(lRes[i]) >= 3:
                clusterCenters.append(KMeans(n_clusters=3, random_state=0).fit(lRes[i]).cluster_centers_)
                yRes.append(y[i])
    else:
        clusterCenters = []
        for i in range(len(lRes)):
            if len(lRes[i]) > 3:
                clusterCenters.append(KMeans(n_clusters=3, random_state=0).fit(lRes[i]).cluster_centers_)
            elif len(lRes[i])==1:
                clusterCenters.append([lRes[i][0],lRes[i][0],lRes[i][0]])
            elif len(lRes[i])==2:
                clusterCenters.append([lRes[i][0],lRes[i][1],lRes[i][1]])
            elif len(lRes[i])==3:
                clusterCenters.append([lRes[i][0],lRes[i][1],lRes[i][2]])
            yRes.append(y[i])

    return clusterCenters, yRes, userIdList


def add_layer(inputs, in_size, out_size, activation_function=None):  # 先不定义激励函数
    # 初始权重为随机变量要比全零要好;矩阵一般开头大写，后面的向量biase一般小写
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 未激活的预测值
    Wx_plus_b = tf.matmul(inputs, Weight) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# def process(tempdata):
#     """
#     处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
#     :return:
#     """
#     with open(path_test) as lines:
#         with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
#             writer = csv.writer(outer)
#             i = 0
#             ret_set = set([])
#             for line in lines:
#                 if i == 0:
#                     i += 1
#                     writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
#                     continue
#                 item = line.split(",")
#                 if item[0] in ret_set:
#                     continue
#                     #
#                 test_line = numpy.vstack((item[3], item[4], item[6], item[7], item[8])).T
#
#                 pred = clf.predict(test_line)[0]
#                 pred = 0 if pred < 0.15 else pred
#                 # if pred != 0 and item[0] in ret_set:
#                 #     for
#                 writer.writerow([item[0], pred])  # 随机值
#                 #
#                 ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重

def writeCsv(test_userIdList, prediction_value):
    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        j = 0
        ret_set = set([])
        for i in range(len(test_userIdList)):
            if j == 0:
                j += 1
                writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                pred = prediction_value[0][0]
                pred = pred if pred > 0.15 else 0
                writer.writerow([test_userIdList[0], pred])  # 随机值
                ret_set.add(test_userIdList[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重
                continue
            if test_userIdList[i] in ret_set:
                continue
            pred = prediction_value[i][0]
            pred = pred if pred>0.15 else 0
            writer.writerow([test_userIdList[i], pred])  # 随机值
            #
            ret_set.add(test_userIdList[i])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重



if __name__ == "__main__":
    start = time.clock()
    path_train = "data/dm/train.csv"  # 训练文件
    path_test = "data/dm/test.csv"  # 测试文件
    path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。
    # PCA提取特征数
    PCA_featureNum = 10

    print("****************** start **********************")
    train_clusterCenters, train_y, train_userIdList = proprocess(path_train)
    test_clusterCenters, test_y, test_userIdList = proprocess(path_test)
    #簇中心合成一个样例
    train_X = []
    test_X = []
    for i in train_clusterCenters:
        train_X.append(numpy.reshape(i, (1, -1)))
    for j in test_clusterCenters:
        test_X.append(numpy.reshape(j, (1, -1)))

    train_X = numpy.reshape(train_X, (-1, 30))
    test_X = numpy.reshape(test_X, (-1, 30))
    train_y = numpy.reshape(train_y, (-1, 1))
    #归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    train_X = min_max_scaler.fit_transform(train_X)
    test_X = min_max_scaler.transform(test_X)
    #PCA
    pca = PCA(n_components=PCA_featureNum)  # 保留×个主成分
    train_X = pca.fit_transform(train_X)  # 把原始训练集映射到主成分组成的子空间中
    test_X = pca.transform(test_X)  # 把原始测试集映射到主成分组成的子空间中
    #训练
    # 定义传入
    xs = tf.placeholder(tf.float32, [None, PCA_featureNum])
    ys = tf.placeholder(tf.float32, [None, 1])
    # 定义输入层1个，隐藏层10个，输出1个神经元的神经网络
    l1 = add_layer(xs, PCA_featureNum, PCA_featureNum+2, activation_function=tf.nn.relu)
    predition = add_layer(l1, PCA_featureNum+2, 1, activation_function=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition)))
    # train_step = tf.train.GradientDescentOptimizer(0.002).minimize(loss)

    train_step =tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                    name='Adam').minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(50):
        sess.run(train_step, feed_dict={xs: train_X, ys: train_y})
        prediction_value = sess.run(predition, feed_dict={xs: test_X})
#        print(prediction_value)

    # 将test_userIdList 和 prediction_value写入
    writeCsv(test_userIdList, prediction_value)
    print('Time used:',time.clock()-start)
