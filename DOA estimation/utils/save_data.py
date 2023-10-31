import numpy as np
import pandas as pd
import copy


def information_save(ori_data_list: list, dir,header=None):
    # data_list : 列表的列表,列表中每一个元素代表一组存储信息
    # 深拷贝重新赋值,避免append改变原始数据
    data_list = copy.deepcopy(ori_data_list)
    for save_data in data_list:
        mean = np.mean(save_data)
        std = np.std(save_data)

        save_data.append(mean)
        save_data.append(std)

    save_data_np = np.array(data_list)
    save_data_pd = pd.DataFrame(save_data_np)
    save_data_pd.to_csv(dir, header=header, index=False)

    return 0