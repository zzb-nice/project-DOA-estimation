import numpy as np
from torch_implement.signals import *
from torch_implement.model import multiLayer_model, Linear_model, multiLayer_model_2
import torch
from torch.optim import SGD
import math

# author-zbb
if __name__ == '__main__':
    # 1.生成数据集
    dataset = ULA_DOA_dataset(if_save_array=False)

    time_point_1 = time.time()
    # 生成两个入射信号的训练数据
    # delta_theta为两个信号入射角度的偏差
    delta_thetas = np.arange(2, 40, step=1)
    for delta_theta in delta_thetas:
        # 生成[-90°,90°]区间内的theta值
        for theta_1 in np.arange(0, 45 - delta_theta, step=1):
            # 对每组角度样本生成多个目标
            for i in range(5):
                # 生成角度带有随机性
                theta = np.concatenate([theta_1 + np.random.rand(1), theta_1 + delta_theta + np.random.rand(1)], axis=0)
                # dataset.Create_DOA_data_ULA(len(theta), theta, snr=10 * np.random.rand(2), snap=512)
                dataset.Create_DOA_data_ULA(len(theta), theta, snr=10 * np.array([1, 1]), snap=512)
    # 标准化输出角度y值
    theta_mean, theta_std = dataset.theta_stdandard()
    time_point_2 = time.time()
    print('time consume:', time_point_2 - time_point_1, sep=' ')

    print(len(dataset), dataset.convariance_matrix[0].shape, dataset.y[0])

    # 2.数据加载器
    data_loader = array_Dataloader(dataset)

    # 3.模型
    net = multiLayer_model_2(132, 2)

    # 4.学习率,优化器,损失函数
    learning_rate = 0.01
    optimizer = SGD(net.parameters(), learning_rate, momentum=0.9, weight_decay=0.0)
    # weight_decay 相当于l2正则项
    loss_function = torch.nn.MSELoss()

    # 5.训练
    epochs = 100
    net.train()
    for epoch in range(epochs):
        train_loss = 0.
        for step, train_data in enumerate(data_loader):
            optimizer.zero_grad()
            input_data, labels = train_data
            output = net(input_data)

            loss = loss_function(output, labels)
            loss.backward()

            optimizer.step()
            # loss.item() 把tensor转化为float
            train_loss += loss.item()

        mean_loss = train_loss / (step + 1)
        print(f'epoch:{epoch}  mean_RMSE_loss:{math.sqrt(mean_loss) * theta_std}')
        # print(f'epoch:{epoch}  mean_loss:{mean_loss}')
