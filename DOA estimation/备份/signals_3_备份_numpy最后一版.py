import numpy as np
import random
import time


# author-zbb
# numpy 版本备份,没加预处理
class ULA_DOA_dataset:
    def __init__(self, if_save_array: bool):
        # 保存协方差矩阵,阵列信号
        self.convariance_matrix = []
        self.if_save_array = if_save_array
        if if_save_array:
            self.array_data_matirx = []

        # 目标值y即为信号对应的theta值
        self.y = []

        # 阵元信息保存在类中
        # 1. 阵元信息(均匀线阵ULA)
        self.M = 12  # 阵元数量
        self.d = 0.1  # 阵元间距(一定要是半波长吗？)
        self.array = np.linspace(0., self.d * (self.M - 1), self.M)

    def Create_DOA_data_ULA(self, num_signal: int, theta: np.ndarray,
                            snr: np.ndarray,
                            snap=512):
        # theta 和 snr 输入必须为list或者np.ndarray,输入list套np.ndarray会报错
        # x(t)=A*s(t)+n(t),先分别生成A,s(t),n(t)
        c = 3 * 10 ** 8  # 光速

        snr = snr  # 信噪比
        theta = theta  # 入射角
        snap = snap  # 快拍数
        assert len(theta) == num_signal, 'error signal setting'
        # f = 3*10**9  # 入射信号频率
        f = 1 / 2 * c / self.d  # 入射信号频率(最高入射频率？)

        lamda = c / f
        # 最终用了随机相位信号,f和lamda没用上

        # 构造矩阵A
        # 通过矩阵乘法得到A
        steer_vector = np.exp(1j * 2 * np.pi * self.array / lamda)  # 生成e^(2pi*j*d_i/lamda)
        horizon_vec = np.exp(1j * np.sin(theta / 180 * np.pi))
        A = steer_vector[:, np.newaxis] @ horizon_vec[np.newaxis, :]

        # 设置输入信号
        signal = np.random.rand(*[num_signal, snap])  # 随机相位
        # 用了snr的广播机制
        signal = np.sqrt(10 ** (snr[:, np.newaxis] / 10)) * np.exp(1j * 2 * np.pi * signal)
        noise = (np.random.randn(*[self.M, snap]) + 1j * np.random.randn(*[self.M, snap])) / np.sqrt(2)

        array_X = A @ signal + noise

        # 将生成数据存入dataset
        self.convariance_matrix.append(self.calculate_convariance_matrix(array_X))
        self.y.append(theta)
        if self.if_save_array:
            self.array_data_matirx.append(array_X)

    def __len__(self):
        return len(self.convariance_matrix)

    def __getitem__(self, item):
        return self.convariance_matrix[item], self.y[item]

    @staticmethod
    def calculate_convariance_matrix(array_x: np.ndarray):
        convariance_matrix = array_x @ (array_x.transpose().conj())
        # 取协方差矩阵的上三角部分,拉伸成一个向量
        # 拉伸成一个向量损失了空间信息
        # 取值时不包含对角线,用np.triu_indices函数
        convariance_vector = convariance_matrix[np.triu_indices(len(array_x), k=1)]
        # # 方式1.输入协方差矩阵的实部和虚部分为两列,保留实部和虚部的对应关系
        # convariance_vector = np.expand_dims(convariance_vector, axis=-1)
        # convariance_vector = np.concatenate([np.real(convariance_vector), np.imag(convariance_vector)], axis=-1)
        # 方式二.实部和虚部直接拼接起来,不保留对应关系
        convariance_vector = np.concatenate([np.real(convariance_vector),np.imag(convariance_vector)],axis=0)

        # 预处理:用l2范数归一化
        # 静态方法中不能使用self,只能通过类来调用另一个静态方法
        convariance_vector = ULA_DOA_dataset.train_convariance_preprocess(convariance_vector)
        return convariance_vector

    @staticmethod
    def train_convariance_preprocess(convariance_vector):
        # 计算向量的l2范数,然后归一化
        l2_norm = np.linalg.norm(convariance_vector,ord=2,keepdims=False)
        return convariance_vector/l2_norm

    def val_convariance_preprocess(self):
        pass


# 一次加载一个batch的内容
class array_Dataloader:
    def __init__(self, dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_data = len(dataset)
        self.index = list(range(self.num_data))
        self.data_count = 0
        if self.shuffle:
            random.shuffle(self.index)

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_count > self.num_data:
            self.data_count = 0
            if self.shuffle:
                random.shuffle(self.index)
            raise StopIteration
        else:
            batch_index = self.index[self.data_count:self.data_count + self.batch_size]
            data_batch = [self.dataset[i] for i in batch_index]
            # 两个返回值时self.dataset[i] 是 tuple 类型
            self.data_count += self.batch_size
            # print(type(data_batch))

            # 添加对batch的处理
            data_batch = tuple(zip(*data_batch))




            return data_batch


def Create_DOA_data_1():
    # x(t)=A*s(t)+n(t),先分别生成A,s(t),n(t)
    # 1. 阵元信息(均匀线阵ULA)
    c = 3 * 10 ** 8  # 光速
    M = 12  # 阵元数量
    d = 0.1  # 阵元间距(一定要是半波长吗？)

    snr = 10  # 信噪比
    snap = 512  # 快拍数
    num_signal = 3  # 入射信号数量
    theta = [-10, 10, 30]  # 信号入射角度
    assert len(theta) == num_signal, 'error signal setting'
    # f = 3*10**9  # 入射信号频率
    f = 1 / 2 * c / d  # 入射信号频率(最高入射频率？)

    lamda = c / f

    # 构造矩阵A
    array = np.linspace(0., d * (M - 1), M)
    # 通过矩阵乘法得到A
    # steer_vector = np.e ** (1j*2*np.pi*array/lamda)
    steer_vector = np.exp(1j * 2 * np.pi * array / lamda)  # 生成e^(2pi*j*d_i/lamda)
    horizon_vec = np.exp(1j * np.sin(np.array(theta) / 180 * np.pi))
    A = steer_vector[:, np.newaxis] @ horizon_vec[np.newaxis, :]

    # 设置输入信号
    signal = np.random.rand(*[num_signal, snap])
    signal = np.sqrt(10 ** (snr / 10)) * np.exp(1j * 2 * np.pi * signal)
    noise = (np.random.randn(*[M, snap]) + 1j * np.random.randn(*[M, snap])) / np.sqrt(2)

    array_X = A @ signal + noise
    return array_X, theta


def Create_DOA_data_2(self, num_signal: int, theta: np.ndarray,
                      snr: np.ndarray,
                      snap=512):
    # theta 和 snr 输入必须为list或者np.ndarray,输入list套np.ndarray会报错
    # x(t)=A*s(t)+n(t),先分别生成A,s(t),n(t)
    c = 3 * 10 ** 8  # 光速

    snr = snr  # 信噪比
    theta = theta  # 入射角
    snap = snap  # 快拍数
    assert len(theta) == num_signal, 'error signal setting'
    # f = 3*10**9  # 入射信号频率
    f = 1 / 2 * c / self.d  # 入射信号频率(最高入射频率？)

    lamda = c / f
    # 最终用了随机相位信号,f和lamda没用上

    # 构造矩阵A
    # 通过矩阵乘法得到A
    steer_vector = np.exp(1j * 2 * np.pi * self.array / lamda)  # 生成e^(2pi*j*d_i/lamda)
    horizon_vec = np.exp(1j * np.sin(theta / 180 * np.pi))
    A = steer_vector[:, np.newaxis] @ horizon_vec[np.newaxis, :]

    # 设置输入信号
    signal = np.random.rand(*[num_signal, snap])  # 随机相位
    # 用了snr的广播机制
    signal = np.sqrt(10 ** (snr[:, np.newaxis] / 10)) * np.exp(1j * 2 * np.pi * signal)
    noise = (np.random.randn(*[self.M, snap]) + 1j * np.random.randn(*[self.M, snap])) / np.sqrt(2)

    array_X = A @ signal + noise

    # 将生成数据存入dataset
    self.convariance_matrix.append(self.calculate_convariance_matrix(array_X))
    self.y.append(theta)
    if self.if_save_array:
        self.array_data_matirx.append(array_X)


if __name__ == '__main__':
    dataset = ULA_DOA_dataset(if_save_array=False)

    # # 测试：生成一个数据
    # theta = [0,50]
    # dataset.Create_DOA_data_ULA(len(theta),theta)
    # print(len(dataset.convariance_matrix),dataset.convariance_matrix[0].shape,dataset.y)

    time_point_1 = time.time()
    # 生成两个入射信号的训练数据
    # delta_theta为两个信号入射角度的偏差
    delta_thetas = np.arange(2, 40, step=2)
    for delta_theta in delta_thetas:
        # 生成[-90°,90°]区间内的theta值

        # for theta_1 in range(-90, 89 - delta_theta, 1):
        #     # 生成角度带有随机性
        #     # random.random()类似与numpy中的np.random.rand()
        #     theta = [theta_1 + random.random(), theta_1 + delta_theta + random.random()]
        #     dataset.Create_DOA_data_ULA(len(theta), np.array(theta), snr=10 * np.random.rand(2), snap=512)
        # 用numpy生成,可调用函数更多,更方便
        for theta_1 in np.arange(-90.0, 89 - delta_theta, step=1):
            # 生成角度带有随机性,不采用整数角
            theta = np.concatenate([theta_1 + np.random.rand(1), theta_1 + delta_theta + np.random.rand(1)],axis=0)
            dataset.Create_DOA_data_ULA(len(theta), theta, snr=10 * np.random.rand(2), snap=512)
    time_point_2 = time.time()
    print('time consume:', time_point_2 - time_point_1, sep=' ')

    print(len(dataset), dataset.convariance_matrix[0].shape, dataset.y)

    dataloader = array_Dataloader(dataset)
    for data_batch in enumerate(dataloader):
        pass