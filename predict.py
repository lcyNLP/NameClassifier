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
n_model = "06"
n_hidden = 128
n_layers = 1

all_letters = string.ascii_letters + '.,;'

n_letters = len(all_letters)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
data_path = "./data/names/"
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize("NFD",s)
                   if unicodedata.category(c) != "Mn"
                   and c in all_letters)

def readLines(filename):
    '''从人中读取每一行加载到内存中'''
    # 打开文件进行读取数据,使用strip去除两侧的空白符，以\n为换行符进行切分
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]

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

def lineToTensor(line):
    '''将人名转换为onehot张量表示,参数lines是输入的r人名'''
    # 首先初始化一个为零的张量,张量的形状是(len(line),l,n_letters)
    # 代表人名中每一个字母都用一个(1 * n_letters)张量表示
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历每个人名列表中的每个字符，并且搜搜到对应的索引,将其索引为1
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return  tensor

n_categories = len(all_categories)

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


def evaluateRNN(line_tensor):
    """评估函数, 和训练函数逻辑相同, 参数是line_tensor代表名字的张量表示"""
    # 初始化隐层张量
    rnn = torch.load("./model/rnn01.pkl")
    hidden = rnn.initHidden()
    # 将评估数据line_tensor的每个字符逐个传入rnn之中
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 获得输出结果
    return output.squeeze(0)



def predict(input_line,  evaluate_fn, n_predictions=3):
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

num_acc_top1 = 0
for index, value in enumerate(data_test.train):
    predictions = predict(value,evaluateRNN, n_predictions=1)
    print("top1:",predictions)
    if predictions[0][1] == data_test.label[index]:
        num_acc_top1 += 1
print("top1准确率为：",num_acc_top1 / data_test.shape[0])


num_acc_top3 = 0
for index, value in enumerate(data_test.train):
    predictions = predict(value,evaluateRNN, n_predictions=3)
    print("top3:",predictions)
    for i in predictions:
        if data_test.label[index] == i[1]:
            num_acc_top3 += 1
print("top3准确率为：",num_acc_top3 / data_test.shape[0])