import numpy as np
import pandas as pd
import xlrd
import openpyxl
from SpatialRegion import SpatialRegion
from transform_poi_class import transform_poi_class
import json


def from_taxi_num_to_time_list(taxi_num_list, taxi_points):
    time_list = []
    for i in range(24):
        time_list.append(0)
    for taxi_num in taxi_num_list:
        time = taxi_points[taxi_num][2]
        if type(time) == int:
            time_list[time] += 1
    return time_list


def from_poi_num_list_to_poi_class_list(poi_num_list, tag):
    poi_class_list = []
    for i in range(25):
        poi_class_list.append(0)
    for poi_num in poi_num_list:
        poi_class = tag[poi_num]
        poi_class_list[poi_class] += 1
    return poi_class_list


def get_vector(x_step, y_step):
    xstep = x_step
    ystep = y_step

    poi = pd.read_excel('shanghai_poi.xlsx', engine='openpyxl')
    poi_loca = poi[['lng', 'lat']]  # 两列分别经/纬
    """索引方式:poi_loca['lng'][0]"""
    print(poi_loca)

    taxi_points = np.load('arrive_point_array.npy', allow_pickle=True)  # 三列分别为经/纬/时
    """索引方式:taxi_points[0][0]"""

    minlat = min(sorted(poi_loca['lat'])[0], sorted(taxi_points[:, 1])[0])
    minlng = min(sorted(poi_loca['lng'])[0], sorted(taxi_points[:, 0])[0])
    maxlat = max(sorted(poi_loca['lat'], reverse=True)[0], sorted(taxi_points[:, 1], reverse=True)[0])
    maxlng = max(sorted(poi_loca['lng'], reverse=True)[0], sorted(taxi_points[:, 0], reverse=True)[0])

    shanghai_spatial_region = SpatialRegion(minlat, minlng, maxlat, maxlng, xstep=xstep, ystep=ystep)

    from_region_to_poi = {}
    from_region_to_taxi = {}

    for i in range(poi_loca.shape[0]):
        if shanghai_spatial_region.coord2cell(poi_loca['lat'][i], poi_loca['lng'][i]) not in from_region_to_poi.keys():
            from_region_to_poi[shanghai_spatial_region.coord2cell(poi_loca['lat'][i], poi_loca['lng'][i])] = []
        else:
            from_region_to_poi[shanghai_spatial_region.coord2cell(poi_loca['lat'][i], poi_loca['lng'][i])].append(i)

    for i in range(taxi_points.shape[0]):
        if shanghai_spatial_region.coord2cell(taxi_points[i][1], taxi_points[i][0]) not in from_region_to_taxi.keys():
            from_region_to_taxi[shanghai_spatial_region.coord2cell(taxi_points[i][1], taxi_points[i][0])] = []
        else:
            from_region_to_taxi[shanghai_spatial_region.coord2cell(taxi_points[i][1], taxi_points[i][0])].append(i)

    # 用poi预测车流数据集的生成
    poi_data = []
    taxi_data = []
    tag, tagz_and_num_match = transform_poi_class()
    for region, poi in from_region_to_poi.items():
        if region not in from_region_to_taxi.keys():
            poi_data.append(from_poi_num_list_to_poi_class_list(poi, tag))
            taxi_data.append(list(np.zeros(24, )))
        else:
            poi_data.append(from_poi_num_list_to_poi_class_list(poi, tag))
            taxi_data.append(from_taxi_num_to_time_list(from_region_to_taxi[region], taxi_points))

    print(poi_data)
    print(taxi_data)
    print(len(poi_data))
    print(len(taxi_data))

    def is_zeros(list):
        for i in range(24):
            if list[i] != 0:
                return False
        return True

    cnt = 0
    for item in taxi_data:
        if is_zeros(item):
            cnt += 1

    print("有poi但没有车流的格子数:")
    print(cnt)

    np.save('poi_data(region=%d%d).npy' % (x_step, y_step), np.array(poi_data))
    np.save('taxi_data(region=%d%d).npy' % (x_step, y_step), np.array(taxi_data))
    with open('tagz_and_num_match(region=%d%d).txt' % (x_step, y_step), "w") as f:
        f.write(json.dumps(tagz_and_num_match, ensure_ascii=False))



