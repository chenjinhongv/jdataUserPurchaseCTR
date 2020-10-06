"""
Coding: utf-8
Author: Jinhong Chan
Date: 2020-09-18 21:20:24
Email: 842960911@qq.com
"""

import numpy as np
import pandas as pd

path = "../data/"


def group_fun(df, groupKey, aggConf, renameConf):
    """
    description:对df进行批量分组聚合操作，返回结果
    param:
        df{pandas.dataFrame}:dataframe wait to be deal
        groupKey{list[str]}:the columns name list which df group by
        aggConf{dict{str:List[aggfunc]}}:agg func setting
        renameConf(List[str]):new column name of the new agg columns
    return:
        res{dataFrame}
    """

    res = df.groupby(groupKey).agg(aggConf)
    res.columns = renameConf
    res = res.reset_index()

    return res


def get_slide_win_action(action, time_column_name, slide_win):
    """
    :param action:
    :param time_column_name:
    :param slide_win:
    :return:
    """
    res = []
    for end_date, span in slide_win:
        label_end_date = pd.Timestamp(end_date)
        label_start_date = label_end_date - pd.Timedelta(days=span)
        label = action.loc[
                (action[time_column_name] > label_start_date) & (action[time_column_name] < label_end_date), :]
        res.append(label)
    return res


def get_action(file="action.h5", start_date="2016-01-31 23:59:02", end_date="2016-04-15 23:59:59", cate=8):
    """
    description:return a dataframe of user's action log
    param：
        file{str}:h5 file name which save user's action log
        start_date{str}:the start_date of user's action that you need with format:"YYYY-MM-DD hh-mm-ss"
        end_date{str}:the end_date of user's action that you need with format:"YYYY-MM-DD hh-mm-ss"
    return:
        res {dataFrame}:user's action log which occur between setted start_date and end_date with columns:
            user_id
            sku_id
            model_id
            type
            cate
            brand
            datetime
            week
            hour
            weekday
    """

    res = pd.read_hdf(path+file, key="action")
    return res.loc[(res.datetime > start_date) & (res.datetime < end_date) & (res.cate == 8), :]


def get_user(file="user.h5"):

    return pd.read_hdf(path+file, key="user")


def get_product(file="product.h5"):

    return pd.read_hdf(path+file, key="product")


def get_comment(file="comment.h5"):

    return pd.read_hdf(path+file, key="comment")


if __name__ == "__main__":
    user = get_user()
    group_key = ["sex"]
    agg_conf = {"user_lv_cd": [np.mean, np.max, np.min], "age_rank": [np.mean, np.std], "user_id": [lambda x:len(x)]}
    rename_conf = ["a", "b", "c", "d", "e", "f"]
    res = groupfun(user, group_key, agg_conf, rename_conf)
    print(res.head(10))
    print(res.columns)

