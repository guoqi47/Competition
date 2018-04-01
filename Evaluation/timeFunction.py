# -*- coding: utf-8 -*-
import time

def processTime(time1, time2):
    time1 = time.strftime("%H", time.localtime(time1))[0:2]
    time2 = time.strftime("%H", time.localtime(time2))[0:2]
    time_list = [0, 0, 0, 0]

    index1 = int((int(time1) - 4) / 6)
    index2 = int((int(time2) - 4) / 6)
    # index1 = int((int(4)-4)/6)
    # index2 = int((int(10)-4)/6)
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
print(processTime(1211,1222))