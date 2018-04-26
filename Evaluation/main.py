# -*- coding:utf8 -*-
import os
import csv
# import pandas as pd
from sklearn.cluster import KMeans
import gc
from math import radians, cos, sin, asin, sqrt, log
import pandas
import numpy
import time
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA
import xgboost as xgb

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


def bo(path):
    min = path.min()
    if min <= 0:
        path = path.map(lambda x: x - min + 0.1)
    pre = path.shift(1)
    change = (pre / path).map(log)
    return change.var()


def haversine(zuo):
    lon1, lat1, lon2, lat2 = map(radians, zuo)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r


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

def proprocess(file_path):
    csv_chunk = pandas.read_csv(file_path, iterator=True,
                                dtype={'TERMINALNO': numpy.int32, 'TIME': numpy.int32,
                                       'TRIP_ID': numpy.int32, 'LONGITUDE': numpy.float64,
                                       'LATITUDE': numpy.float64,
                                       'Y': numpy.float64, 'DIRECTION': numpy.float64,
                                       'HEIGHT': numpy.float64, 'SPEED': numpy.float64,
                                       'CALLSTATE': numpy.int32})

    df = pandas.DataFrame(columns=['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'Y',
                                   'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE'], index=[0])

    curr_id = 0
    user_index = [[], []]
    new_row = []
    y_list = []
    tmp_user = pandas.DataFrame(columns=['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'Y',
                                         'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE'], index=[0])

    try:
        while True:
            df2 = csv_chunk.get_chunk(10000)
            flag = True
            df = pandas.concat([tmp_user, df2])
            while flag:
                trip_id = 1
                line_num = 0
                user = df.ix[df['TERMINALNO'] == curr_id]
                if not user.empty:
                    y_list.append([curr_id, user.iat[0, 9]])
                    user = user.sort_values(by='TIME')
                    user = user.reset_index(drop=True)
                    user['pre'] = user['TIME'].shift(1)
                    user['cha'] = user['TIME'] - user['pre']
                    gap = user.ix[user['cha'] >= 1800]
                    for index, row in gap.iterrows():
                        trip = user.ix[line_num:index - 1]
                        le = len(trip) - 1
                        line_num = index
                        new_row.append([trip.SPEED, trip.DIRECTION, trip.HEIGHT,
                                        [trip.iat[0, 4], trip.iat[0, 3], trip.iat[le, 4], trip.iat[le, 3]],
                                        (trip.iat[0, 7], trip.iat[-1, 7]), trip.iat[0, 0],
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                        user_index[0].append(curr_id)
                        user_index[1].append(trip_id)
                        trip_id += 1
                    trip = user.ix[line_num:]
                    le = len(trip) - 1
                    new_row.append([trip.SPEED, trip.DIRECTION, trip.HEIGHT,
                                    [trip.iat[0, 4], trip.iat[0, 3], trip.iat[le, 4], trip.iat[le, 3]],
                                    (trip.iat[0, 7], trip.iat[-1, 7]), trip.iat[0, 0],
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    user_index[0].append(curr_id)
                    user_index[1].append(trip_id)
                    curr_id += 1
                else:
                    curr_id -= 1
                    tmp_user = df.ix[df['TERMINALNO'] == curr_id]
                    flag = False
    except StopIteration:
        print('Stop read')
    tuples = list(zip(*user_index))
    index = pandas.MultiIndex.from_tuples(tuples, names=['user_id', 'trip_id'])
    user_trips = pandas.DataFrame(new_row, columns=['speeds', 'dir', 'hig', 'coor',
                                                    'time', 'call', "dis", 't1', 't2', 't3', 't4', 'time_list',
                                                    'var_dir', 'ave_v', 'var_v', 'ave_h', 'std_v', 'std_dir', 'std_h',
                                                    'skew_h', 'skew_v', 'kurt_h', 'kurt_v', 'coe_var_h', 'coe_var_v',
                                                    'mean_v', 'var_h', 'bo_h', 'bo_v'
                                                    ],
                                  index=index)
    user_trips['var_dir'] = user_trips['dir'].map(lambda x: x.var())
    user_trips['std_dir'] = user_trips['dir'].map(lambda x: x.std())
    user_trips.pop('dir')
    # 以下速度均由官方SPEED一栏计算
    user_trips['var_v'] = user_trips['speeds'].map(lambda x: x.var())
    user_trips['mean_v'] = user_trips['speeds'].map(lambda x: x.mean())
    user_trips['std_v'] = user_trips['speeds'].map(lambda x: x.std())
    user_trips['kurt_v'] = user_trips['speeds'].map(lambda x: x.kurt())
    user_trips['skew_v'] = user_trips['speeds'].map(lambda x: x.skew())
    user_trips['bo_v'] = user_trips['speeds'].map(bo)
    user_trips['coe_var_v'] = (user_trips['std_v'] / user_trips['mean_v'])
    user_trips.pop('speeds')

    user_trips['ave_h'] = user_trips['hig'].map(lambda x: x.mean())
    user_trips['var_h'] = user_trips['hig'].map(lambda x: x.var())
    user_trips['std_h'] = user_trips['hig'].map(lambda x: x.std())
    user_trips['skew_h'] = user_trips['hig'].map(lambda x: x.skew())
    user_trips['kurt_h'] = user_trips['hig'].map(lambda x: x.kurt())
    user_trips['coe_var_h'] = (user_trips['std_h'] / user_trips['ave_h'])
    user_trips['bo_h'] = user_trips['hig'].map(bo)
    user_trips.pop('hig')
    user_trips['dis'] = user_trips['coor'].map(haversine)
    user_trips.pop('coor')
    # 该平均速度为起点和终点直线距离与时间的比值
    user_trips['ave_v'] = user_trips['dis'] / user_trips['time'].map(lambda x: (x[1] - x[0]) / (60.0 * 60))
    user_trips['time_list'] = user_trips['time'].map(time_list)
    user_trips.pop('time')
    user_trips['t1'] = user_trips['time_list'].map(lambda x: x[0])
    user_trips['t2'] = user_trips['time_list'].map(lambda x: x[1])
    user_trips['t3'] = user_trips['time_list'].map(lambda x: x[2])
    user_trips['t4'] = user_trips['time_list'].map(lambda x: x[3])
    user_trips.pop('time_list')
    user_trips.fillna(0, inplace=True)

    Y_data = pandas.DataFrame(y_list, columns=['id', 'Y'])
    Y_data.drop_duplicates(subset='id', inplace=True)
    Y_data.set_index('id', inplace=True)
    del df
    del new_row
    del y_list
    gc.collect()
    print(time.time() - t1)
    #    return user_trips, Y_data
    user_trips.fillna(0) #空值填充0


    userIdList = []
    y = []
    userId = 0
    Y_index = 0
#    y.append(Y_data['Y'][Y_index])
#    Y_index += 1
#    userIdList.append(userId)
    l = []
    lRes = []  # 最后按user_id分的列表
    yRes = []  # 最后按user_id分的列表
    # userIdListRes =
#    a=[]
#    for i in range(len(Y_data)):
#       a.append(user_trips.query('index[0]=='+str(i)))
    count=0
    for index, row in user_trips.iterrows():
        if index[0] == userId:
            row = numpy.where((row.isnull() ) | (row>10000000),0,row)
            l.append(row)

            if count == len(user_trips) - 1:
                lRes.append(l)
                userIdList.append(index[0])
                y.append(Y_data['Y'][Y_index])
        else:
            lRes.append(l)
            userIdList.append(userId)
            y.append(Y_data['Y'][Y_index])
            Y_index += 1

            l = []
            row = numpy.where((row.isnull() ) | (row>10000000),0,row)
            l.append(row)
            userId = index[0]
        count+=1

    # 进行聚类处理
    if file_path == path_train:
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
            elif len(lRes[i]) == 1:
                clusterCenters.append([lRes[i][0], lRes[i][0], lRes[i][0]])
            elif len(lRes[i]) == 2:
                clusterCenters.append([lRes[i][0], lRes[i][1], lRes[i][1]])
            elif len(lRes[i]) == 3:
                clusterCenters.append([lRes[i][0], lRes[i][1], lRes[i][2]])
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
                pred = prediction_value[0]
                writer.writerow([test_userIdList[0], pred])  # 随机值
                ret_set.add(test_userIdList[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重
                continue
            if test_userIdList[i] in ret_set:
                continue
            pred = prediction_value[i]
            writer.writerow([test_userIdList[i], pred])  # 随机值
            #
            ret_set.add(test_userIdList[i])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重
#        writer.writerow([test_userIdList[-1], prediction_value[-1]])  # 随机值


if __name__ == "__main__":
    start = time.clock()
    path_train = "data/dm/train.csv"  # 训练文件
    path_test = "data/dm/test.csv"  # 测试文件
    path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。
    # PCA提取特征数
    PCA_featureNum = 30

    print("****************** start **********************")
    train_clusterCenters, train_y, train_userIdList = proprocess(path_train)
    print('***read train set', time.clock() - start)
    test_clusterCenters, test_y, test_userIdList = proprocess(path_test)
    print('***read test set', time.clock() - start)
    print('***Neural network Begin:', time.clock() - start)
    # 簇中心合成一个样例
    train_X = []
    test_X = []
    for i in train_clusterCenters:
        train_X.append(numpy.reshape(i, (1, -1)))
    for j in test_clusterCenters:
        test_X.append(numpy.reshape(j, (1, -1)))

    train_X = numpy.reshape(train_X, (-1, 69))
    test_X = numpy.reshape(test_X, (-1, 69))
    train_y = numpy.reshape(train_y, (-1, 1))
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    train_X = min_max_scaler.fit_transform(train_X)
    test_X = min_max_scaler.transform(test_X)
#    # PCA
#    pca = PCA(n_components=PCA_featureNum)  # 保留×个主成分
#    train_X = pca.fit_transform(train_X)  # 把原始训练集映射到主成分组成的子空间中
#    test_X = pca.transform(test_X)  # 把原始测试集映射到主成分组成的子空间中
#    # 训练,定义传入
#    xs = tf.placeholder(tf.float32, [None, PCA_featureNum])
#    ys = tf.placeholder(tf.float32, [None, 1])
#    # 定义输入层1个，隐藏层10个，输出1个神经元的神经网络
#    l1 = add_layer(xs, PCA_featureNum, PCA_featureNum + 2, activation_function=tf.nn.relu)
#    predition = add_layer(l1, PCA_featureNum + 2, 1, activation_function=None)
#    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition)))
#    # train_step = tf.train.GradientDescentOptimizer(0.002).minimize(loss)
#
#    train_step = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
#                                        name='Adam').minimize(loss)
#
#    init = tf.global_variables_initializer()
#    sess = tf.Session()
#    sess.run(init)
#    for i in range(1000):
#        sess.run(train_step, feed_dict={xs: train_X, ys: train_y})
#        prediction_value = sess.run(predition, feed_dict={xs: test_X})
#    #        print(prediction_value)

#     XGBoost训练过程
    #基于Scikit-learn接口
    model = xgb.XGBRegressor(max_depth=3,
                             learning_rate=0.1,
                             n_estimators=200,
                             silent=True,
                             objective='reg:linear',
                             booster='gbtree',
                             
                             gamma=0,
                             min_child_weight=3,
                             max_delta_step=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             reg_alpha=1,
                             reg_lambda=1,
                             scale_pos_weight=1,
                             base_score=0.5,
                             seed=47)
    model.fit(train_X, train_y)

    # 对测试集进行预测
    ans = model.predict(test_X)

    #原生接口
#    params = {
#        'booster': 'gbtree',
#        'objective': 'reg:linear',
#        'gamma': 0,
#        'max_depth': 6,
#        'lambda': 5,
#        'subsample': 0.8,
#        'colsample_bytree': 0.7,
#        'min_child_weight': 1.5,
#        'silent': 1,
#        'eta': 0.01,
#        'seed': 100,
#    }
#    dtrain = xgb.DMatrix(train_X, train_y)
#    num_rounds = 1000
#    plst = params.items()
#    model = xgb.train(plst, dtrain, num_rounds)
#
#    # 对测试集进行预测
#    dtest = xgb.DMatrix(test_X)
#    ans = model.predict(dtest)
    # 将test_userIdList 和 prediction_value写入
    writeCsv(test_userIdList, ans)
    print('Time used:', time.clock() - start)
