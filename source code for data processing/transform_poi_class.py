import numpy as np
import pandas as pd


def unique(list):
    new_list = []
    for i in range(len(list)):
        if list[i] not in new_list:
            new_list.append(list[i])
    return new_list


def list_to_numlist(list):
    list_unique = unique(list)
    list_unique_len = len(list_unique)
    num_list = []
    for i in list:
        for j in range(list_unique_len):
            if list_unique[j] == i:
                num_list.append(j)
    return num_list


def transform_poi_class():
    poi = pd.read_excel('shanghai_poi.xlsx', engine='openpyxl')
    poi_tagz = poi['tagz']
    tag = list_to_numlist(poi_tagz)
    tagz_and_num_match = {}
    for i in range(len(tag)):
        if poi_tagz[i] not in tagz_and_num_match.keys():
            tagz_and_num_match[poi_tagz[i]] = tag[i]
    return tag, tagz_and_num_match





