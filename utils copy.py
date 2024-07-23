import time
from datetime import datetime
import pandas as pd

def parse_time(time_string):
    '''
        返回/月/日格式的时间的timestamp格式
        
        time_string 格式为 %Y/%m/%d
    '''
    return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())


def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    '''
        feature_number
        1   关注close
        2   关注close、volume
        3   关注close、high、low
        4   关注close、high、low、open
        其他值均为非法值
    '''
    if feature_number == 1:
        type_list = ["close"]
    elif feature_number == 2:
        type_list = ["close", "volume"]
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["close", "high", "low"]
    elif feature_number == 4:
        type_list = ["close", "high", "low", "open"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list


def get_volume_forward(time_span, portion, portion_reversed):
    volume_forward = 0
    # 如果portion_reversed不等于 None, False, 空字符串”", 0, 空列表[], 空字典{}, 空元组()时
    # volume_forward 赋值为 
    if not portion_reversed:
        volume_forward = time_span*portion
    return volume_forward


def panel_fillna(df, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    '''
        填充缺失值
        bfill()函数用于向后填充缺失值，即用后面的非缺失值来填充当前的缺失值；ffill()函数用于向前填充缺失值，即用前面的非缺失值来填充当前的缺失值。
    '''
    frames = {}
    for item in df.items:
        if type == "both":
            frames[item] = df.loc[item].fillna(axis=1, method="bfill").\
                fillna(axis=1, method="ffill")
        else:
            frames[item] = df.loc[item].fillna(axis=1, method=type)
    return pd.DataFrame(frames)
