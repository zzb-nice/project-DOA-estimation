import numpy as np
import torch
import random
import time


# author-zbb
# torch 版本
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
        self.M = 8  # 阵元数量
        self.d = 0.1  # 阵元间距(一定要是半波长吗？)
        self.array = np.linspace(0., self.d * (self.M - 1), self.M)

    def Create_DOA_data_ULA(self, num_signal: int, theta: np.ndarray,
                            snr: np.ndarray,
                            snap=512):
        # theta 和 snr 输入必须为np.ndarray,输入list或list套np.ndarray会报错
        # x(t)=A*s(t)+n(t),先分别生成A,s(t),n(t)
        c = 3 * 10 ** 8  # 光速

        snr = snr  # 信噪比
        theta = theta  # 入射角
        snap = snap  # 快拍数
        assert len(theta) == num_signal, 'error signal setting'
        # f = 3*10**9  # 入射信号频率
        f = 1 / 2 * c / self.d  # 入射信号频率(最高入射频率？)

        lamda = c / f
        # 用随机相位信号作为输入s(t)时,f和lamda没用上

        # 构造矩阵A
        # 通过矩阵乘法得到A
        # 他喵,这也写错啦
        steer_vector = 1j * 2 * np.pi * self.array / lamda  # 生成e^(2pi*j*d_i/lamda)
        horizon_vec = np.sin(theta / 180 * np.pi)
        A = np.exp(steer_vector[:, np.newaxis] @ horizon_vec[np.newaxis, :])

        # 设置输入信号
        # signal = np.random.rand(*[num_signal, snap])  # 随机相位
        # 错错
        # signal = np.ones([num_signal, snap])
        # 时域采样频率怎么设置?
        signal_time = np.linspace(0, snap - 1, snap) / (2 * snap)
        signal_time = np.expand_dims(signal_time, 0).repeat(num_signal, axis=0)
        # 加一个初始相位
        signal_time = signal_time + np.random.rand(num_signal, 1) / 2

        # 用了snr的广播机制
        signal = np.sqrt(10 ** (snr[:, np.newaxis] / 10)) * np.exp(1j * 2 * np.pi * f * signal_time)
        # 随机相位
        # signal = np.sqrt(10 ** (snr / 10)) * np.exp(1j * 2 * np.pi * signal)
        noise = (np.random.randn(*[self.M, snap]) + 1j * np.random.randn(*[self.M, snap])) / np.sqrt(2)

        array_X = A @ signal + noise

        # 将生成数据存入dataset
        self.convariance_matrix.append(self.calculate_convariance_matrix(array_X))
        # 输入和目标都改成float32型,减小计算量
        self.y.append(theta.astype(np.float32))
        if self.if_save_array:
            self.array_data_matirx.append(array_X.astype(np.float32))

    def theta_stdandard(self):
        y = np.array(self.y)
        y_mean = np.mean(y)
        y_std = np.std(y)

        y = (y - y_mean) / y_std
        # 注意最终要转换回列表类型
        self.y = list(y)
        return y_mean, y_std

    def __len__(self):
        return len(self.convariance_matrix)

    def __getitem__(self, item):
        return self.convariance_matrix[item], self.y[item]

    @staticmethod
    def calculate_convariance_matrix(array_x: np.ndarray):
        convariance_matrix = array_x @ (array_x.transpose().conj())
        snap = array_x.shape[-1]
        convariance_matrix = convariance_matrix / snap
        # 后续加上画convariance_matrix图的函数
        # convariance_matrix只有对角线是实数,应该没问题
        # 算出来有点大。。忘记除snap了
        # 取协方差矩阵的上三角部分,拉伸成一个向量
        # 拉伸成一个向量损失了空间信息
        # 取值时不包含对角线,用np.triu_indices函数
        convariance_vector = convariance_matrix[np.triu_indices(len(array_x), k=1)]
        # # 方式1.输入协方差矩阵的实部和虚部分为两列,保留实部和虚部的对应关系
        # convariance_vector = np.expand_dims(convariance_vector, axis=-1)
        # convariance_vector = np.concatenate([np.real(convariance_vector), np.imag(convariance_vector)], axis=-1)
        # 方式二.实部和虚部直接拼接起来,不保留对应关系
        convariance_vector = np.concatenate([np.real(convariance_vector), np.imag(convariance_vector)], axis=0)

        # 预处理:用l2范数归一化
        # 静态方法中不能使用self,只能通过类来调用另一个静态方法
        convariance_vector = ULA_DOA_dataset.train_convariance_preprocess(convariance_vector)

        # 计算完成后最终转换成float32类型
        convariance_vector = convariance_vector.astype(np.float32)
        return convariance_vector

    @staticmethod
    def train_convariance_preprocess(convariance_vector):
        # 计算向量的l2范数,然后归一化
        l2_norm = np.linalg.norm(convariance_vector, ord=2, keepdims=False)
        return convariance_vector / l2_norm

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

            # torch处理
            input_data, labels = data_batch

            input_data = torch.as_tensor(np.array(input_data))
            labels = torch.as_tensor(np.array(labels))
            # input_data = torch.stack(input_data, dim=0)

            return input_data, labels


def Create_one_signal(dataset: ULA_DOA_dataset, snr=-10, snap=128):
    # 单个入射信号
    # 生成0-45°对应的数据
    # 共产生45*15=675组数据
    start, end = 0, 45
    # np.arange 不包含end,包含end-1
    theta_range = np.arange(start, end, step=1)
    # 一组角度样本产生若干组数据
    repeat_array = 15
    time_point_1 = time.time()
    print('add single angle signal data...')
    for theta in theta_range:
        for i in range(repeat_array):
            # 生成的角度带有随机性
            # 把信噪比snr自动转化为np.ndarray
            # np.array([1])用np.ons更方便
            dataset.Create_DOA_data_ULA(1, theta + np.random.rand(1), snr=snr * np.ones(1), snap=snap)

    time_point_2 = time.time()
    print(f'time Consume:{time_point_2 - time_point_1}', end='  ')
    print(f'{len(theta_range)}*{repeat_array}={len(theta_range) * repeat_array} data has been created')


def Create_two_signal(dataset: ULA_DOA_dataset, snr=-10, snap=128):
    # 两个入射信号
    # 生成0-45°对应的数据
    # 共产生 *5= 组数据
    start, end = 0, 45
    # 数据间隔
    delta_thetas_1 = np.array([2, 3, 4, 8, 12, 16, 20, 24, 30])
    # 一组角度样本产生若干组数据
    repeat_array = 5
    time_point_1 = time.time()
    print('add two angles signal data...')
    count = 0  # 计数产生的数据量
    # 遍历所有间隔数据
    for theta_i in delta_thetas_1:
            # 在数据间隔能实行时才加入数据
            if start + theta_i <= end - 1:
                # np.arange 不包含end,包含end-1
                for theta in np.arange(start, end - theta_i, step=1):
                    # 生成的角度带有随机性
                    # 把信噪比snr自动转化为np.ndarray
                    # theta = np.array([theta,theta+theta_i,theta+theta_i+theta_j])  错误,代码只支持一维np.adarray形式
                    # 这样可以,theta截取之后是0维
                    theta = np.array([theta, theta + theta_i])
                    for i in range(repeat_array):
                        # 每组角度样本产生多组数据
                        dataset.Create_DOA_data_ULA(2, theta + np.random.rand(2), snr=snr * np.ones(2), snap=snap)

                # 计数
                count += len(np.arange(start, end - theta_i, step=1))

    time_point_2 = time.time()
    print(f'time Consume:{time_point_2 - time_point_1}', end='  ')
    print(f'{count}*{repeat_array}={count * repeat_array} data has been created')


def Create_three_signal(dataset: ULA_DOA_dataset,repeat_array=3, snr=-10, snap=128):
    # 多个入射信号
    # 生成0-45°对应的数据
    # 共产生 *5= 组数据
    start, end = 0, 45
    # 数据间隔
    delta_thetas_1 = np.array([2, 3, 4, 8, 12, 16, 20, 24, 30])
    delta_thetas_2 = np.array([2, 3, 4, 8, 12, 16, 20, 24, 30])
    # 一组角度样本产生若干组数据
    repeat_array = repeat_array
    time_point_1 = time.time()
    print('add three angles signal data...')
    count = 0  # 计数产生的数据量
    # 遍历所有间隔数据
    for theta_i in delta_thetas_1:
        for theta_j in delta_thetas_2:
            # 在数据间隔能实行时才加入数据
            if start + theta_i + theta_j <= end - 1:
                # np.arange 不包含end,包含end-1
                for theta in np.arange(start, end - theta_i - theta_j, step=1):
                    # 生成的角度带有随机性
                    # 把信噪比snr自动转化为np.ndarray
                    # theta = np.array([theta,theta+theta_i,theta+theta_i+theta_j])  错误,代码只支持一维np.adarray形式
                    # 这样可以,theta截取之后是0维
                    theta = np.array([theta, theta + theta_i, theta + theta_i + theta_j])
                    for i in range(repeat_array):
                        # 每组角度样本产生多组数据
                        dataset.Create_DOA_data_ULA(3, theta + np.random.rand(3), snr=snr * np.ones(3), snap=snap)

                # 计数
                count += len(np.arange(start, end - theta_i - theta_j, step=1))

    time_point_2 = time.time()
    print(f'time Consume:{time_point_2 - time_point_1}', end='  ')
    print(f'{count}*{repeat_array}={count * repeat_array} data has been created')


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
            theta = np.concatenate([theta_1 + np.random.rand(1), theta_1 + delta_theta + np.random.rand(1)], axis=0)
            dataset.Create_DOA_data_ULA(len(theta), theta, snr=10 * np.random.rand(2), snap=512)
    time_point_2 = time.time()
    print('time consume:', time_point_2 - time_point_1, sep=' ')

    print(len(dataset), dataset.convariance_matrix[0].shape, dataset.y)

    dataloader = array_Dataloader(dataset)
    for data_batch in enumerate(dataloader):
        pass
