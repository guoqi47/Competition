# coding:utf-8

import gc
from math import radians, cos, sin, asin, sqrt
import pandas
import numpy
import time
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


def compress(file_path):
    csv_chunk = pandas.read_csv(file_path, iterator=True,
                                dtype={'TERMINALNO': numpy.int32, 'TIME': numpy.int32,
                                       'TRIP_ID': numpy.int32, 'LONGITUDE': numpy.float64,
                                       'LATITUDE': numpy.float64,
                                       'Y': numpy.float64, 'DIRECTION': numpy.float64,
                                       'HEIGHT': numpy.float64, 'SPEED': numpy.float64,
                                       'CALLSTATE': numpy.int32})


    n = 0
    df = pandas.DataFrame(columns=['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'Y',
                                   'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE'], index=[0])

    curr_id = 1
    new_row = []
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
                    curr_id -= 1
                    tmp_user = df.ix[df['TERMINALNO'] == curr_id]
                    flag = False

    except StopIteration:
        print('Stop read')
        pass

    user_trips = pandas.DataFrame(new_row, columns=['user_id', 'trip_id', 'speeds', 'dir', 'hig', 'coor',
                                                    'time', 'Y', 'call', "dis", 't1', 't2', 't3', 't4', 'time_list',
                                                    'var_dir', 'ave_v', 'var_v', 'ave_h'])
    user_trips['var_dir'] = user_trips['dir'].map(lambda x: x.var()).map(lambda x: x if pandas.notnull(x) else 0)
    user_trips.pop('dir')
    user_trips['var_v'] = user_trips['speeds'].map(lambda x: x.var())
    user_trips.pop('speeds')
    user_trips['ave_h'] = user_trips['hig'].map(lambda x: x.mean())
    user_trips.pop('hig')
    user_trips['dis'] = user_trips['coor'].map(haversine)
    user_trips.pop('coor')
    user_trips['ave_v'] = (user_trips['dis'] / user_trips['time'].map(lambda x: x[1] - x[0]))
    user_trips['time_list'] = user_trips['time'].map(time_list)
    user_trips.pop('time')
    user_trips['t1'] = user_trips['time_list'].map(lambda x: x[0])
    user_trips['t2'] = user_trips['time_list'].map(lambda x: x[1])
    user_trips['t3'] = user_trips['time_list'].map(lambda x: x[2])
    user_trips['t4'] = user_trips['time_list'].map(lambda x: x[3])
    user_trips.pop('time_list')
    user_trips.fillna(0, inplace=True)
    del df
    gc.collect()
    print(time.time() - t1)
    return user_trips, curr_id


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    train = compress('data/dm/555.csv')[0]
    train.to_csv('co.csv')

