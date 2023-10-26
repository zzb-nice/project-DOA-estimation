import numpy as np
from numpy_implement.signals_numpy import *

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error

# author-zbb
if __name__ == '__main__':
    # 1.生成训练集
    dataset = ULA_DOA_dataset(if_save_array=False)

    time_point_1 = time.time()
    # 生成两个入射信号的训练数据
    # delta_theta为两个信号入射角度的偏差
    delta_thetas = np.arange(1, 40, step=1)
    for delta_theta in delta_thetas:
        # 生成[-90°,90°]区间内的theta值
        for theta_1 in np.arange(0, 45 - delta_theta, step=1):
            # 生成角度带有随机性
            theta = np.concatenate([theta_1 + np.random.rand(1), theta_1 + delta_theta + np.random.rand(1)], axis=0)
            # dataset.Create_DOA_data_ULA(len(theta), theta, snr=10 * np.random.rand(2), snap=512)
            dataset.Create_DOA_data_ULA(len(theta), theta, snr=30 * np.array([1, 1]), snap=512)
    # 标准化输出角度y值
    theta_mean, theta_std = dataset.theta_stdandard()
    time_point_2 = time.time()
    print('time consume:', time_point_2 - time_point_1, sep=' ')

    print(len(dataset), dataset.convariance_matrix[0].shape, dataset.y[0])

    # 2.生成验证集
    val_dataset = ULA_DOA_dataset(if_save_array=False)

    time_point_1 = time.time()
    # 生成两个入射信号的训练数据
    # delta_theta为两个信号入射角度的偏差
    delta_thetas = np.arange(1, 40, step=2)
    for delta_theta in delta_thetas:
        # 生成[-90°,90°]区间内的theta值
        for theta_1 in np.arange(0, 45 - delta_theta, step=1):
            # 生成角度带有随机性
            theta = np.concatenate([theta_1 + np.random.rand(1), theta_1 + delta_theta + np.random.rand(1)], axis=0)
            # dataset.Create_DOA_data_ULA(len(theta), theta, snr=10 * np.random.rand(2), snap=512)
            val_dataset.Create_DOA_data_ULA(len(theta), theta, snr=30 * np.array([1, 1]), snap=512)
    # 标准化输出角度y值,验证集不保存对应的均值和方差
    val_dataset.theta_stdandard()
    time_point_2 = time.time()
    print('time consume:', time_point_2 - time_point_1, sep=' ')

    # 调用sklearn的模型
    # model = LinearRegression()
    # model = RandomForestRegressor()
    # model = svm.SVR()
    # model = svm.SVR(kernel='rbf', C=1.)
    model = svm.SVR(kernel='poly', C=1.,degree=5)
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=0, loss='ls')
    X, y = np.array(dataset.convariance_matrix), np.array(dataset.y)[:,0]
    model.fit(X, y)

    y_hat = model.predict(X)

    # loss = np.sum((y - y_hat).transpose() @ (y - y_hat)) / (y.shape[0] * y.shape[1])  # 写错啦,狠狠记住！
    # loss = np.sum((y - y_hat) * (y - y_hat)) / (y.shape[0] * y.shape[1])
    loss = mean_squared_error(y, y_hat,squared=False)
    print('train_RMSE:',loss * theta_std * theta_std)

    X, y = np.array(val_dataset.convariance_matrix), np.array(val_dataset.y)[:,0]
    model.fit(X, y)

    y_hat = model.predict(X)
    loss = mean_squared_error(y, y_hat,squared=False)
    print('val_RMSE:',loss * theta_std * theta_std)
