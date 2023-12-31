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

    # 生成数据
    Create_one_signal(dataset, snr=-10, snap=128)

    # 标准化输出角度y值
    theta_mean, theta_std = dataset.theta_stdandard()

    print(len(dataset), dataset.convariance_matrix[0].shape, dataset.y[0])

    # 2.生成验证集
    val_dataset = ULA_DOA_dataset(if_save_array=False)

    Create_one_signal(val_dataset,snr=-10,snap=128)

    # 标准化输出角度y值,验证集不保存对应的均值和方差
    val_dataset.theta_stdandard()


    # 调用sklearn的模型
    # model = LinearRegression()
    # model = RandomForestRegressor()
    # model = svm.SVR(kernel='linear', C=10.)
    model = svm.SVR(kernel='rbf', C=10.)
    # model = svm.SVR(kernel='poly', C=1.,degree=5)
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
    X, y = np.array(dataset.convariance_matrix), np.array(dataset.y)
    model.fit(X, y)

    y_hat = model.predict(X)

    # loss = np.sum((y - y_hat).transpose() @ (y - y_hat)) / (y.shape[0] * y.shape[1])  # 写错啦,狠狠记住！
    # loss = np.sum((y - y_hat) * (y - y_hat)) / (y.shape[0] * y.shape[1])
    loss = mean_squared_error(y, y_hat, squared=False)

    # MSE:乘theta_std * theta_std,RMSE乘theta_std
    print('train_RMSE:', loss * theta_std)

    X, y = np.array(val_dataset.convariance_matrix), np.array(val_dataset.y)

    y_hat = model.predict(X)
    loss = mean_squared_error(y, y_hat, squared=False)
    print('val_RMSE:', loss * theta_std)
