"""
Coding: utf-8
Author: Jinhong Chan
Date: 2020-09-18 21:20:59
Email: 842960911@qq.com
"""

from utils import *
from functools import reduce
import os
import re


def get_sample(action):
    """
    :param action:
    :return:user_id x sku_id which act in given action and have not buy yet
    """
    buy_uxp = action.loc[action["type"] == 4, :].drop_duplicates(subset=["user_id", "sku_id"]).\
                  loc[:, ["user_id", "sku_id"]]
    act_uxp = action.drop_duplicates(subset=["user_id", "sku_id"]).loc[:, ["user_id", "sku_id"]]
    pd.concat([buy_uxp, act_uxp]).drop_duplicates(subset=["user_id", "sku_id"],keep=False)
    return pd.concat([buy_uxp, act_uxp]).drop_duplicates(subset=["user_id", "sku_id"], keep=False)


def get_label(action):
    res = action.loc[action["type"] == 4, ["user_id","sku_id"]].drop_duplicates(subset=["user_id","sku_id"])
    res["label"] = 1
    return res


def get_user_base_feat():
    base_feat = ["user_id", "sex", "user_lv_cd", "age_rank", "reg_age"]
    return get_user().loc[:, base_feat]


def get_product_base_feat():
    base_feat = ["sku_id", "a1", "a2", "a3", "cate", "brand", "comment_num",
                 "has_bad_comment", "bad_comment_rate"]

    res = pd.merge(get_product(), get_comment(), how="left", on=["sku_id"])
    return res.loc[:, base_feat]


def get_user_acc_action_feat(action, end_date, path):
    """
    :TODO:测试
    :param action:
    :param end_date:
    :return:
    """
    file_name = "user_acc_action_feat_{}.csv".format(re.sub(r"\D", "", end_date))
    if os.path.exists(path+file_name):
        res = pd.read_csv(path+file_name)
    else:
        action["day_of_year"] = action["datetime"].apply(lambda x: x.dayofyear)
        action = pd.concat([action, pd.get_dummies(action["type"], prefix="action_type")], axis=1)

        # 活动天数，最近交互时间
        agg_conf = {
            "day_of_year":["nunique"],
            "datetime": [lambda x: (pd.Timestamp(end_date) - max(x)).days]
        }
        rename_conf = ["user_active_days", "user_last_active_time"]
        df = action.loc[:, ["user_id", "day_of_year", "datetime"]]
        com_feat = group_fun(df, ["user_id"], agg_conf, rename_conf)

        # 购买天数，最近购买时间
        agg_conf = {
            "day_of_year":["nunique"],
            "datetime": [lambda x: (pd.Timestamp(end_date) - max(x)).days],
        }
        rename_conf = ["user_buy_days", "user_last_buy_time"]
        df = action.loc[:, ["user_id", "day_of_year", "datetime"]]
        com_buy_feat = group_fun(df, ["user_id"], agg_conf, rename_conf)

        # 用户对商品各种行为计数的四种常规统计量，最多行为数，最少行为数，标准差，平均行为数
        uxp_action_count = action[["user_id", "sku_id"]+["action_type_"+str(x) for x in range(1,7)]].\
            groupby(["user_id", "sku_id"]).sum().reset_index().drop(columns=["sku_id"], axis=1)
        agg_conf = dict(zip(list(uxp_action_count.columns)[1:], [[np.max, np.min,np.std, np.average]]*6))
        rename_conf = reduce(lambda x, y: x+y, [["uxp_"+x+"_max", "uxp_"+x+"_min", "uxp_"+x+"_std", "uxp_"+x+"_avg"] for x in uxp_action_count.columns[1:]])
        uxp_action_static_feat = group_fun(uxp_action_count, ["user_id"], agg_conf, rename_conf)
        del uxp_action_count

        # 用户对商品各种行为计数的四种常规统计量，最多行为数，最少行为数，标准差，平均行为数
        uxb_action_count = action[["user_id","brand"]+["action_type_"+str(x) for x in range(1,7)]].\
            groupby(["user_id","brand"]).sum().reset_index().drop(columns=["brand"],axis=1)
        agg_conf = dict(zip(list(uxb_action_count.columns)[1:],[[np.max,np.min,np.std,np.average]]*6))
        rename_conf = reduce(lambda x,y:x+y,[["uxb_"+x+"_max","uxb_"+x+"_min","uxb_"+x+"_std", "uxb_"+x+"_avg"] for x in uxb_action_count.columns[1:]])
        uxb_action_static_feat = group_fun(uxb_action_count, ["user_id"], agg_conf, rename_conf)
        del uxb_action_count

        # 各种行为计数
        u_action_feat = action[["user_id"]+["action_type_"+str(x) for x in range(1,7)]].\
            groupby(["user_id"]).sum()
        u_action_feat.rename(columns=dict(zip(list(u_action_feat.columns), [x + "_count" for x in list(u_action_feat.columns)])), inplace=True)
        u_action_feat = pd.concat([u_action_feat, action[["user_id","type"]].groupby(["user_id"]).count()], axis=1)
        u_action_feat.rename(columns={"type":"action_count_all"},inplace=True)
        # 各种行为转化率
        u_action_feat["action_type_1_ratio"] = u_action_feat["action_type_4_count"]/u_action_feat["action_type_1_count"]
        u_action_feat["action_type_2_ratio"] = u_action_feat["action_type_4_count"]/u_action_feat["action_type_2_count"]
        u_action_feat["action_type_3_ratio"] = u_action_feat["action_type_4_count"]/u_action_feat["action_type_3_count"]
        u_action_feat["action_type_5_ratio"] = u_action_feat["action_type_4_count"]/u_action_feat["action_type_5_count"]
        u_action_feat["action_type_6_ratio"] = u_action_feat["action_type_4_count"]/u_action_feat["action_type_6_count"]
        u_action_feat.reset_index(inplace=True)

        res = pd.merge(com_feat,com_buy_feat,on=["user_id"],how="left")
        res = pd.merge(res,uxp_action_static_feat,on=["user_id"],how="left")
        res = pd.merge(res,uxb_action_static_feat,on=["user_id"],how="left")
        res = pd.merge(res,u_action_feat,on=["user_id"],how="left")

        res.to_csv(path+file_name, index=False)

    return res


def get_uxp_action_acc_feat(action, end_date, path):
    file_name = "uxp_action_acc_feat_{}.csv".format(re.sub(r"\D", "", end_date))
    if os.path.exists(path+file_name):
        res = pd.read_csv(path+file_name)
    else:
        action["day_of_year"] = action["datetime"].apply(lambda x: x.dayofyear)
        action = pd.concat([action, pd.get_dummies(action["type"], prefix="action_type")], axis=1)

        # 用户对商品各种行为计数
        uxp_action_count_feat = action[["user_id", "sku_id"]+["action_type_"+str(x) for x in range(1, 7)]].\
            groupby(["user_id", "sku_id"]).sum()
        uxp_action_count_feat.rename(columns=dict(zip(list(uxp_action_count_feat.columns), ["uxp" + x + "_count" for x in list(uxp_action_count_feat.columns)])), inplace=True)
        uxp_action_count_feat.reset_index(inplace=True)

        # 用户对商品最后交互时间和交互天数
        agg_conf = {
            "day_of_year": ["nunique"],
            "datetime": [lambda x: (pd.Timestamp(end_date) - max(x)).days]
        }
        rename_conf = ["uxp_active_days", "uxp_last_active_time"]
        df = action.loc[:, ["user_id", "sku_id", "day_of_year", "datetime"]]
        uxp_action_com_feat = group_fun(df, ["user_id", "sku_id"], agg_conf, rename_conf)

        res = pd.merge(uxp_action_count_feat, uxp_action_com_feat, on=["user_id", "sku_id"], how="left")
        res.to_csv(path+file_name, index=False)

    return res


def get_product_action_acc_feat(action, end_date, path):
    file_name = "product_action_acc_feat_{}.csv".format(re.sub(r"\D", "", end_date))
    if os.path.exists(path+file_name):
        res = pd.read_csv(path+file_name)
    else:
        action["day_of_year"] = action["datetime"].apply(lambda x: x.dayofyear)
        action = pd.concat([action, pd.get_dummies(action["type"], prefix="action_type")], axis=1)
        # 活动天数，活动总量，活动人数
        agg_conf = {
            "day_of_year":["nunique"],
            "type":["count"],
            "user_id":["nunique"]
        }
        rename_conf = ["p_action_days","p_action_count","p_user_count"]
        p_action_com_feat = group_fun(action,["sku_id"],agg_conf,rename_conf)

        # 购买天数，购买总量，购买人数
        rename_conf = ["p_buy_action_days","p_buy_action_count","p_buy_user_count"]
        p_buy_action_com_feat = group_fun(action,["sku_id"],agg_conf,rename_conf)

        # 来自用户的行为数四种统计量，最大值，最小值，平均值，标准差
        uxp_action_count = action[["user_id","sku_id"]+["action_type_"+str(x) for x in range(1,7)]].\
            groupby(["user_id","sku_id"]).sum().reset_index().drop(columns=["user_id"],axis=1)
        agg_conf = dict(zip(list(uxp_action_count.columns)[1:],[[np.max,np.min,np.std,np.average]]*6))
        rename_conf = reduce(lambda x,y:x+y,[["uxp_"+x+"_max","uxp_"+x+"_min","uxp_"+x+"_std","uxp_"+x+"_avg"] for x in uxp_action_count.columns[1:]])
        p_user_action_static_feat = group_fun(uxp_action_count, ["sku_id"], agg_conf, rename_conf)
        del uxp_action_count

        res = pd.merge(p_action_com_feat,p_buy_action_com_feat,on=["sku_id"],how="left")
        res = pd.merge(res,p_user_action_static_feat,on=["sku_id"],how="left")
        res.to_csv(path+file_name,index=False)

    return res


def get_uxb_action_acc_feat(action, end_date, path):

    # 用户对品牌的各种行为统计

    # 用户对品牌的交互天数

    # 用户对品牌的商品购买数量
    pass


def get_user_action_feat_win(action, end_date, span, path):
    pass


def get_product_action_feat_win(action, end_date, span, path):
    pass


if __name__ == "__main__":
    action = get_action()
    user_act_feat = get_user_acc_action_feat(action.loc[action.datetime < "2016-04-01 23:59:59", :], "2016-04-01 23:59:59",
                                             "../cache/")
