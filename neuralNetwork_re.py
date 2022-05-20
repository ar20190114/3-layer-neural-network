import numpy
import scipy.special
import matplotlib.pyplot as plt


class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # ノード数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 行列の計算式
        # w11 w21
        # w12 w22 etc
        # P.160
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 学習率
        self.lr = learningrate

        # シグモイド関数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #学習
    def train(self, inputs_list, targets_list):
        # 入力値と目標値の転置
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隠れ層の入力値
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隠れ層の出力値
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層の入力値
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 出力層の出力値
        final_outputs = self.activation_function(final_inputs)

        # 目標値と出力層の出力値の誤差
        output_errors = targets - final_outputs

        # 重みと誤差の行列計算
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # P.95
        # 降下勾配法
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # 降下勾配法
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    #学習ででた重みを用いる
    def query(self, inputs_list):
        # 入力値
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 隠れ層の入力値
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隠れ層の出力値
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層の入力値
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 出力層の出力値
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# ノード数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# 学習率
learning_rate = 0.1

# ニューラルネットワーク
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# CSVファイルの読み込み
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# ニューラルネットワークの学習

# エポック数
epochs = 20
E = []


for e in range(epochs):

    for record in training_data_list:
        all_values = record.split(',')
        # 入力値のスケーリング
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 目標配列の生成
        targets = numpy.zeros(output_nodes) + 0.01

        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    #pass

    # CSVファイルの読み込み
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # ニューラルネットワークの実行

    # 正解スコアのリスト
    scorecard = []


    for record in test_data_list:

        all_values = record.split(',')
        # 正解ラベル
        correct_label = int(all_values[0])
        #　入力値のスケーリング
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # ネットワークへの照会
        outputs = n.query(inputs)
        # 最大値のインデックスがラベルに対応
        label = numpy.argmax(outputs)
        # リストへの追加
        if (label == correct_label):
            # 正解なら1
            scorecard.append(1)
        else:
            # 間違いなら0
            scorecard.append(0)
            pass

        pass

    print(n)

    # 正答率
    scorecard_array = numpy.asarray(scorecard)
    E.append(scorecard_array.sum() / scorecard_array.size)
    print(e)

    pass
print(E)

x = numpy.linspace(0, 19, 20)
plt.plot(x, E)
plt.show()
