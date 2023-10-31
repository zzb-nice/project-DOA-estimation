import numpy as np
from numpy_implement.signals_numpy import *
from utils.save_data import information_save

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def train_test_one():
    # 固定快拍数为10,信噪比从-25变化到-15
    snap = 10
    snrs = np.linspace(-25, -15, 5)

    # 存储各信噪比情况下的训练,测试损失
    total_train_loss = []
    total_val_loss = []
    # 1.生成训练集
    for step,snr in enumerate(snrs):
        dataset = ULA_DOA_dataset(if_save_array=False)

        Create_three_signal(dataset, repeat_array=3, snr=snr, snap=snap)
        theta_mean, theta_std = dataset.theta_stdandard()

        val_dataset = ULA_DOA_dataset(if_save_array=False)

        Create_three_signal(val_dataset, repeat_array=3, snr=snr, snap=snap)
        val_dataset.theta_stdandard()

        # 2.调用sklearn的模型
        # model = LinearRegression()
        # model = RandomForestRegressor()
        # model = svm.SVR()
        model = svm.SVR(kernel='rbf', C=1.)
        # model = svm.SVR(kernel='poly', C=1.,degree=5)
        # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=0)  # , loss='ls'
        model = MultiOutputRegressor(model)
        X, y = np.array(dataset.convariance_matrix), np.array(dataset.y)
        model.fit(X, y)

        y_hat = model.predict(X)

        # 还原输出的归一化,目标和预测值都需要还原
        y_hat = y_hat * theta_std + theta_mean
        y = y * theta_std + theta_mean

        train_loss = mean_squared_error(y, y_hat, squared=False)
        print(step,'.','train_RMSE:', train_loss)

        X, y = np.array(val_dataset.convariance_matrix), np.array(val_dataset.y)
        model.fit(X, y)

        y_hat = model.predict(X)

        y_hat = y_hat * theta_std + theta_mean
        y = y * theta_std + theta_mean

        val_loss = mean_squared_error(y, y_hat, squared=False)
        print(step,'.','val_RMSE:', val_loss)

        # 3.保存相应的数据
        # train_loss 为shape:()的numpy数组
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)

    # snrs 在画图的时候作为横坐标,在表格的时候作为表头
    return total_train_loss, total_val_loss, snrs


def train_test_two():
    # 固定信噪比为-15db,快拍数按列表变化
    snaps = [1, 5, 10, 15, 20, 30]
    snr = -15

    # 存储各信噪比情况下的训练,测试损失
    total_train_loss = []
    total_val_loss = []
    # 1.生成训练集
    for step,snap in enumerate(snaps):
        dataset = ULA_DOA_dataset(if_save_array=False)

        Create_three_signal(dataset, repeat_array=3, snr=snr, snap=snap)
        theta_mean, theta_std = dataset.theta_stdandard()

        val_dataset = ULA_DOA_dataset(if_save_array=False)

        Create_three_signal(val_dataset, repeat_array=3, snr=snr, snap=snap)
        val_dataset.theta_stdandard()

        # 2.调用sklearn的模型
        # model = LinearRegression()
        # model = RandomForestRegressor()
        # model = svm.SVR()
        model = svm.SVR(kernel='rbf', C=1.)
        # model = svm.SVR(kernel='poly', C=1.,degree=5)
        # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=0)  # , loss='ls'
        model = MultiOutputRegressor(model)
        X, y = np.array(dataset.convariance_matrix), np.array(dataset.y)
        model.fit(X, y)

        y_hat = model.predict(X)

        # 还原输出的归一化,目标和预测值都需要还原
        y_hat = y_hat * theta_std + theta_mean
        y = y * theta_std + theta_mean

        train_loss = mean_squared_error(y, y_hat, squared=False)
        print(step,'.','train_RMSE:', train_loss)

        X, y = np.array(val_dataset.convariance_matrix), np.array(val_dataset.y)
        model.fit(X, y)

        y_hat = model.predict(X)

        y_hat = y_hat * theta_std + theta_mean
        y = y * theta_std + theta_mean

        val_loss = mean_squared_error(y, y_hat, squared=False)
        print(step,'.','val_RMSE:', val_loss)

        # 3.保存相应的数据
        # train_loss 为shape:()的numpy数组
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)

    # snaps 在画图的时候作为横坐标,在表格的时候作为表头
    return total_train_loss, total_val_loss, snaps


# author-zbb
if __name__ == '__main__':
    # 1.生成训练集
    dataset = ULA_DOA_dataset(if_save_array=False)

    Create_three_signal(dataset, repeat_array=3, snr=-10, snap=128)
    theta_mean, theta_std = dataset.theta_stdandard()

    val_dataset = ULA_DOA_dataset(if_save_array=False)

    Create_three_signal(val_dataset, repeat_array=3, snr=-10, snap=128)
    val_dataset.theta_stdandard()

    # 调用sklearn的模型
    # model = LinearRegression()
    # model = RandomForestRegressor()
    # model = svm.SVR()
    model = svm.SVR(kernel='rbf', C=1.)
    # model = svm.SVR(kernel='poly', C=1.,degree=5)
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=0)  # , loss='ls'
    model = MultiOutputRegressor(model)
    X, y = np.array(dataset.convariance_matrix), np.array(dataset.y)
    model.fit(X, y)

    y_hat = model.predict(X)

    # 还原输出的归一化,目标和预测值都需要还原
    y_hat = y_hat * theta_std + theta_mean
    y = y * theta_std + theta_mean
    # loss = np.sum((y - y_hat).transpose() @ (y - y_hat)) / (y.shape[0] * y.shape[1])  # 写错啦,狠狠记住！
    # loss = np.sum((y - y_hat) * (y - y_hat)) / (y.shape[0] * y.shape[1])
    loss = mean_squared_error(y, y_hat, squared=False)
    print('train_RMSE:', loss)

    X, y = np.array(val_dataset.convariance_matrix), np.array(val_dataset.y)
    model.fit(X, y)

    y_hat = model.predict(X)

    y_hat = y_hat * theta_std + theta_mean
    y = y * theta_std + theta_mean

    loss = mean_squared_error(y, y_hat, squared=False)
    print('val_RMSE:', loss)
