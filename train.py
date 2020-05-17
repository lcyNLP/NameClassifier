# 从io中导入文件打开方法
from io import open
# 帮助使用正则表达式进行子目录的查询
import glob
import os
# 用于获得常见字母及字符规范化
import string
import unicodedata
# 导入随机工具random
import random
# 导入时间和数学工具包
import time
import math
# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
# 引入制图工具包
import matplotlib.pyplot as plt

all_letters = string.ascii_letters + '.,;'

n_letters = len(all_letters)
# print("n_letters:", n_letters)

# 关于编码问题我们暂且不去考虑
# 我们认为这个函数的作用就是去掉一些语言中的重音标记
# 如: Ślusàrski ---> Slusarski
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize("NFD",s)
                   if unicodedata.category(c) != "Mn"
                   and c in all_letters)

# s = "Ślusàrski"
# a = unicodeToAscii(s)
# print(a)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
data_path = "./data/names/"
def readLines(filename):
    '''从人中读取每一行加载到内存中'''
    # 打开文件进行读取数据,使用strip去除两侧的空白符，以\n为换行符进行切分
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]

# filename = data_path + "Chinese.txt"
# reslut = readLines(filename)
# print(reslut[:20])

# 构建的category_lines形如：{"English":["Lily", "Susan", "Kobe"], "Chinese":["Zhang San", "Xiao Ming"]}
category_lines = {}

# 构建所有类别的列表all_categories形如： ["English",...,"Chinese"]
all_categories = []

# 遍历所有的文件,使用glob中的的正则表达式的遍历
for filename in glob.glob(data_path + "*.txt"):
    # 获取每个文件的文件名,其实就是得到名字的类别
    caregory = os.path.splitext(os.path.basename(filename))[0]
    # 逐一的将其装入所有类别中
    all_categories.append(caregory)
    # 然后读取每个文件的内容，形成名字的列表
    lines = readLines(filename)
    # 按照对应的列表,讲名字列表写入到category_lines字典中
    category_lines[caregory] = lines

n_categories = len(all_categories)



def lineToTensor(line):
    '''将人名转换为onehot张量表示,参数lines是输入的r人名'''
    # 首先初始化一个为零的张量,张量的形状是(len(line),l,n_letters)
    # 代表人名中每一个字母都用一个(1 * n_letters)张量表示
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历每个人名列表中的每个字符，并且搜搜到对应的索引,将其索引为1
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return  tensor

n_layers = 1
#第三步构建rnn模型

class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers = n_layers):
        '''
        RNN网络的初始化
        :param input_size: 代表RNN输入的最后一个维度
        :param hidden_size: 代表RNN隐层的最后一个维度
        :param output_size: 代表RNN网络最后线性层的输出维度
        :param num_layers: 代表RNN的网络层数
        '''
        super(RNN, self).__init__()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.RNN, 它的三个参数分别是input_size, hidden_size, num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden):
        """完成传统RNN中的主要逻辑, 输入参数
        input代表输入张量, 它的形状是1 x n_letters
        hidden代表RNN的隐层张量, 它的形状是self.num_layers * 1 * self.hidden_size
        """
        # 因为预定义的nn.RNN要求输入维度一定是三维张量, 因此在这里使用unsqueeze(0)扩展一个维度
        input1 = input1.unsqueeze(0).to(device)
        # 讲input1和hidden输入到RNN的实例化对象中
        # 如果num_layers=1，rr恒等于hn
        rr, hn = self.rnn(input1, hidden.to(device))
        # 将从RNN中获得结果通过线性层的变换和softmax的处理，最终返回结果
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        '''
        用来初始化一个全零的隐藏层的张量
        :return:
        '''
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(device)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = n_layers):
        '''
        LSTM初始化
        :param input_size:代表输入张量x最后一个维度
        :param hidden_size:代表隐藏层张量的最后一个维度
        :param output_size:代表线性层最后的输出维度
        :param num_layers:代表LSTM代表网络的层数
        '''
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden, c):
        '''
        完成LSTM的主要逻辑,网络的输入有三个张量
        :param input1:输入
        :param hidden:隐藏层
        :param c:细胞状态
        :return:
        '''
        # input1 = input1.unsqueeze(0)
        input1 = input1.unsqueeze(0).to(device)

        # 讲三个参数输入到LSTM对象中
        rr, (hn, cn)= self.lstm(input1, (hidden.to(device), c.to(device)))
        # 将最后三个张量结果全部返回，同事rr要经过线性层和softmax
        return self.softmax(self.linear(rr)), hn, cn

    def initHiddenAndC(self):
        """初始化函数不仅初始化hidden还要初始化细胞状态c, 它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden.to(device), c.to(device)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=n_layers):
        '''
        模型的初始化
        :param input_size:代表输入张量x的最后一个维度
        :param hidden_size:代表隐藏层最后一个维度
        :param output_size:代表指定线性层输出的维度
        :param num_layers:代表GRU网络的层数
        '''
        super(GRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.GRU, 它的三个参数分别是input_size, hidden_size, num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        # 实例化线性层的对象
        self.linner = nn.Linear(hidden_size, output_size)
        # 定义softmax对象,作用是从输出张量中的到类别的分类
        # 在最后一个维度上作用softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden):
        input1 = input1.unsqueeze(0).to(device)
        rr, hn = self.gru(input1,hidden.to(device))
        return self.softmax(self.linner(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(device)



# 因为是onehot编码, 输入张量最后一维的尺寸就是n_letters
input_size = n_letters

# 定义隐层的最后一维尺寸大小
n_hidden = 128

# 输出尺寸为语言类别总数n_categories
output_size = n_categories

# num_layer使用默认值, num_layers = 1

# 假如我们以一个字母B作为RNN的首次输入, 它通过lineToTensor转为张量
# 因为我们的lineToTensor输出是三维张量, 而RNN类需要的二维张量
# 因此需要使用squeeze(0)降低一个维度
input = lineToTensor('B').squeeze(0)
print("input.shape",input.shape)

# 初始化一个三维的隐层0张量, 也是初始的细胞状态张量
hidden = c = torch.zeros(1, 1, n_hidden)

rnn = RNN(n_letters, n_hidden, n_categories).to(device)
lstm = LSTM(n_letters, n_hidden, n_categories).to(device)
gru = GRU(n_letters, n_hidden, n_categories).to(device)


# 第四步构建训练函数
def categoryFromOutput(output):
    """从输出结果中获得指定类别, 参数为输出张量output"""
    # output:充输出结果中得到指定的类别
    # 从输出张量中返回最大的值和索引对象, 我们这里主要需要这个索引
    top_n, top_i = output.topk(1)
    # top_i对象中取出索引的值
    category_i = top_i[0].item()
    # 根据索引值获得对应语言类别, 返回语言类别和索引值
    return all_categories[category_i], category_i



# 随机生成训练数据
def randomTrainingExample():
    # 该函数的作用：用于随机产生训练数据
    # 第一步使用random.choice（）方法从all_categories中随机选择一个类别
    caregory = random.choice(all_categories)
    # 第二步通过category_lines字典中取出category类别对应的名字列表
    line = random.choice(category_lines[caregory])
    # 第三步讲类别封装成tensor
    category_tensor = torch.tensor([all_categories.index(caregory)],dtype=torch.long)
    # 讲随机渠道的名字通过函数linToTensor转换乘one-hot张量
    line_tensor = lineToTensor(line)
    return caregory, line, category_tensor.to(device), line_tensor.to(device)



# 定义损失函数nn.NLLLoss，因为RNN的最后一层为nn.LogSoftmax，两者的内部计算逻辑正好吻合
criterion = nn.NLLLoss()

# 设置学习率
learning_rate = 0.03

def trainRNN(category_tensor, line_tensor):
    '''
    定义训练函数
    :param category_tensor:类别的张量，相当于训练数据的标签
    :param line_tensor: 名字的张量表示，相当于训练数据
    :return: output, loss.item()
    '''
    # 第一步初始化rnn隐藏层的张量
    optimizer = torch.optim.SGD(rnn.parameters(),lr=learning_rate,momentum=0.9)

    hidden = rnn.initHidden().to(device)

    # 关键的一步，将模型结构中的梯度归零
    optimizer.zero_grad()
    # optimizer.zero_grad()

    # 循环遍历训练数据中line_tensor的每一个字符，传入rnn中，并且迭代更新hidden
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 因为rnn输出的是三维张量，为了满足category_tensor，需要进行降维处理
    loss = criterion(output.squeeze(0), category_tensor)
    # 进行反向传播
    loss.backward()
    optimizer.step()

    return output, loss.item()

optimizer_lstm = torch.optim.SGD(lstm.parameters(), lr=learning_rate, momentum=0.9)
# 构建LSTM训练模型
def trainLSTM(category_tensor, line_tensor):
    '''
    实现LSTM的训练函数
    :param category_tensor: 类别的张量，相当于标签label
    :param line_tensor: 名字的张量相当于训练数据
    :return: output, loss
    '''
    # 初始化隐藏层张量，以及初始化细胞状态

    hidden, c = lstm.initHiddenAndC()
    optimizer_lstm.zero_grad()
    for i in range(line_tensor.size()[0]):
        # 进行模型的训练
        output, hidden, c = lstm(line_tensor[i], hidden.to(device), c.to(device))

    # 损失计算，因为output是三维张量因此需要降维
    loss = criterion(output.squeeze(0), category_tensor)
    # 进行反向传播
    loss.backward()
    optimizer_lstm.step()

    return output, loss.item()

# 构建GRU训练模型
def trainGRU(category_tensor, line_tensor):
    '''
    构建gru的训练函数
    :param category_tensor:标签
    :param line_tensor: 特征
    :return:
    '''
    optimizer = torch.optim.SGD(gru.parameters(), lr=learning_rate, momentum=0.9)
    hidden = gru.initHidden().to(device)
    # 梯度清零
    optimizer.zero_grad()
    # 迭代循环训练
    for i in range(line_tensor.size()[0]):
        output, hideen = gru(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

# 构建时间计数函数
def timeSince(since):
    '''
    获取训练的时间
    :param since:训练开始的时间
    :return:
    '''
    # 获取当前时间
    now = time.time()
    # 获取时间差,就是训练耗时
    s = now - since
    # 讲秒转化为分钟, 并且取整
    m = math.floor(s / 60)
    # 计算剩下不够的凑成一分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

since = time.time() - 10 *60
period = timeSince(since)
print(period)


# 训练过程,以及日志的打印
# 设置训练迭代次数
n_iters = 500000
# 设置结果打印间隔
print_every = 10000
# 设置绘制损失曲线图上的之徒间隔
plot_every = 5000
# 设置计算准确率的批次数

def train(train_type_fn):
    '''
    训练过程的日志打印函数
    :param train_type_fn:代表train_type_fn代表选择哪种训练函数
    :return:
    '''
    # 每个制图间隔损失保存列表
    all_losses = []
    all_train_acc = []
    # 获取训练的开始时间
    start = time.time()
    # 设置初始间隔损失为0
    current_loss = 0
    # 谁知初始准确率0
    current_acc = 0
    # 从1开始训练迭代，总n_iters次
    for iter in range(1, n_iters+1):
        # 通过randomTrainExample()来俗急的获取一组训练数据和标签
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # 讲训练特征和标签张量传入训练函数中进行模型的训练
        output, loss = train_type_fn(category_tensor, line_tensor)
        # 累加损失值
        current_loss += loss
        # 取该迭代步骤output通过函数categoryFromOutput()获取对应的类别和索引
        guess, guess_i = categoryFromOutput(output)
        current_acc += 1 if guess == category else 0

        # 如果到了迭代次数打印间隔
        if iter % print_every == 0:
            # 判断和真实类别标签进行比较，如果相同则为True，如果不同则为false
            correct = '✓' if guess == category else '✗ (%s)' % category
            
            # 打印迭代步, 迭代步百分比, 当前训练耗时, 损失, 该步预测的名字, 以及是否正确
            print('%d %d%% (%s) %.4f %s / %s %s||acc:%.4f' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct, current_acc/print_every))
            all_train_acc.append(current_acc/print_every)
            current_acc = 0

        # 如果到了迭代次数的制图间隔
        if iter % plot_every == 0:
            # 将过去若干轮的平均顺势添加到all_loss列表中
            all_losses.append(current_loss / plot_every)
            # 讲损失重置为0
            current_loss = 0
    # 返回训练的总损失,和训练的时间
    return all_losses, all_train_acc,int(time.time()-start)

# 调用train函数，分别进行RNN， LSTM， GRU模型的训练
# 并返回各自的全部损失，以及训练耗时，用于制图
# print("-----------RNN-------------")
# all_losses1,all_train_acc1, period1 = train(trainRNN)
# torch.save(rnn,"./model/rnn05.pkl")
print("-----------LSTM------------")
all_losses2, all_train_acc2, period2 = train(trainLSTM)
torch.save(lstm.state_dict(),"./model/lstm15.1.pt")
torch.save(optimizer_lstm.state_dict(),"./model/optimizer_lstm15.1.pt")
# print("-----------GRU-------------")
# all_losses3, all_train_acc3, period3 = train(trainGRU)
# torch.save(gru,"./model/gru05.pkl")


# 绘制损失对比曲线, 训练耗时对比柱张图
# 创建画布0
plt.figure(0)
# 绘制损失对比曲线
# plt.plot(all_losses1, label="RNN")
plt.plot(all_losses2, color="red", label="LSTM")
# plt.plot(all_losses3, color="orange", label="GRU")
plt.legend(loc='upper left')
plt.savefig("./plt/15.1-loss.png")
# 创建画布1
plt.figure(1)
# 绘制准确率对比曲线
# plt.plot(all_train_acc1, label="RNN")
plt.plot(all_train_acc2, color="red", label="LSTM")
# plt.plot(all_train_acc3, color="orange", label="GRU")
plt.legend(loc='upper left')
plt.savefig("./plt/15.1-acc.png")

# # 创建画布2
# plt.figure(2)
# x_data=["RNN", "LSTM", "GRU"]
# # y_data = [period1, period2, period3]
# # 绘制训练耗时对比柱状图
# # plt.bar(range(len(x_data)), y_data, tick_label=x_data)
# plt.savefig("./plt/05-time.png")
plt.show()


def evaluateRNN(line_tensor):
    """评估函数, 和训练函数逻辑相同, 参数是line_tensor代表名字的张量表示"""
    # 初始化隐层张量
    rnn = RNN(n_letters, n_hidden, n_categories).to(device)
    rnn.load_state_dict(torch.load("./model/rnn15.1.pt"))
    hidden = rnn.initHidden()
    # 将评估数据line_tensor的每个字符逐个传入rnn之中
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 获得输出结果
    return output.squeeze(0)

def evaluateLSTM(line_tensor):
    # 初始化隐层张量和细胞状态张量
    lstm = LSTM(n_letters, n_hidden, n_categories).to(device)
    lstm.load_state_dict(torch.load("./model/lstm15.1.pt"))
    hidden, c = lstm.initHiddenAndC()
    # 将评估数据line_tensor的每个字符逐个传入lstm之中
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    return output.squeeze(0)

def evaluateGRU(line_tensor):
    gru = GRU(n_letters, n_hidden, n_categories).to(device)
    gru.load_state_dict(torch.load("./model/gru15.1.pt"))
    hidden = gru.initHidden()
    # 将评估数据line_tensor的每个字符逐个传入gru之中
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    return output.squeeze(0)


def predict(input_line,evaluate_fn, n_predictions=3):
    '''
    预测函数
    :param input_line:输入的名字
    :param n_predictions: 最有可能的n_predictions个
    :return:
    '''
    # 以下操作的相关张量不进行求梯度
    with torch.no_grad():
        # 使输入的名字转换为张量表示, 并使用evaluate函数获得预测输出
        output = evaluate_fn(lineToTensor(input_line))

        # 从预测的输出中取前3个最大的值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 创建盛装结果的列表
        predictions = []
        # 遍历n_predictions
        for i in range(n_predictions):
            # 从topv中取出的output值
            value = topv[0][i].item()
            # 取出索引并找到对应的类别
            category_index = topi[0][i].item()
            # 打印ouput的值, 和对应的类别
            # print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果装进predictions中
            predictions.append([value, all_categories[category_index]])
    return predictions

import pandas as pd
data_test = pd.read_csv("./data/test_100.csv",names=["label","train"])

# print(predict("LiuChenYang",evaluateLSTM))

num_acc_top1 = 0

for index, value in enumerate(data_test.train):
    predictions = predict(value,evaluateLSTM, n_predictions=1)
    print("top1:",predictions)
    if predictions[0][1] == data_test.label[index]:
        num_acc_top1 += 1

num_acc_top3 = 0
for index, value in enumerate(data_test.train):
    print("value", value)
    predictions = predict(value,evaluateLSTM, n_predictions=3)
    print("top3:", predictions)
    for i in predictions:
        if data_test.label[index] == i[1]:
            num_acc_top3 += 1
print("top1准确率为：",num_acc_top1 / data_test.shape[0])
print("top3准确率为：",num_acc_top3 / data_test.shape[0])
