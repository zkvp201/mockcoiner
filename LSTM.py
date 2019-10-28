import sys
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import pybithumb
from pybithumb import Bithumb
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


tf.set_random_seed(0)  # 그래프 연산에 사용하는 seed 값

# Hyeperparameters
data_dim = 5
hidden_dim = 10
output_dim = 1
seq_length = 60
learing_rate = 0.01
N_EPOCH = 1300
LSTM_stack = 1

# data를 0~1사이의 값으로 변환(Normalization)
def minMaxScaler(data):
    numerator = data - np.min(data, axis=0) # 분자
    denominator = np.max(data, 0) - np.min(data, 0) + 1e-8  # 분모
    return numerator / denominator

# 정규화하기전의 값과 되돌리고 싶은 값을 입력하면 역정규화된 값을 리턴
def reverse_minMaxScaler(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-8)) + org_x_np.min()

# Load data
df = pybithumb.get_ohlcv("BTC")
pred_label = 3 # 0 : Open, 1: High 2: Low, 3:Close 4:Volume

dataset_temp = df.to_numpy()
dataset = minMaxScaler(dataset_temp)

test_min = np.min(dataset_temp, 0)
test_max = np.max(dataset_temp, 0)
test_denom = test_max - test_min

# 문자열 data를 float로 변환
stock_info = df.values[1:].astype(np.float) 

# Preprocessing

# price 데이터 정규화
# 전체 column에서 마지막(volume)을 제외한 모든 열까지 slice. 가격 data를 만든다.
price = stock_info[:, :-1]
norm_price = minMaxScaler(price) # 가격 데이터 (x, 4) 정규화

# volume 데이터 정규화
volume = stock_info[:, -1:]
norm_volume = minMaxScaler(volume)

x = np.concatenate((norm_price, norm_volume), axis=1) # 열 합치기
y = x[:, [-2]] # y(target)는 'Close'

# training/test dataset
dataX = [] # Sequence Data for input
dataY = [] # for target

for i in range(len(y) - seq_length):
    _x = x[i : i+seq_length]
    _y = y[i+seq_length]
    dataX.append(_x)
    dataY.append(_y)

# train:test = 7:3
train_size = int( 0.7 * len(dataY) )
test_size = len(dataY) - train_size

# train data, test data
trainX, testX = np.array( dataX[:train_size] ), np.array( dataX[train_size:len(dataX)] )
trainY, testY = np.array( dataY[:train_size] ), np.array( dataY[train_size:len(dataY)] )

# input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim], name='input_X')
Y = tf.placeholder(tf.float32, [None, 1], name='intput_Y')

# build a LSTM network
def lstm_cell():
    cell_temp = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, 
        state_is_tuple=True, 
        activation=tf.nn.softsign
    )
    return cell_temp

cell = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(LSTM_stack)], 
    state_is_tuple=True
)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# output은 Fully Connected layer를 거친다
hypothesis = tf.contrib.layers.fully_connected(
    outputs[:, -1], 
    output_dim, 
    activation_fn=None
)  

# loss function
loss = tf.reduce_sum(tf.square(hypothesis - Y), name='sumOfSE')  # sum of the square Error

# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss, name='train')

# 검증용 측정지표를 산출하기 위한 tagrets, predictions 생성
targets = tf.placeholder(tf.float32, [None, 1], name='targets')
predictions = tf.placeholder(tf.float32, [None, 1], name='predictions')
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)), name='RMSE')

train_error_List = [] # 학습 데이터 오차 기록
test_error_List = []  # 테스트 데이터 오차 기록
test_predict = ''

# ======================== 세션을 열고 학습 시작 =============================================
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training step
for epoch in range(N_EPOCH):
    _, step_loss = sess.run(
        [train, loss], feed_dict={X: trainX, Y: trainY}
    )
    n_display = 100
    if ( (epoch+1) % n_display == 0 ) : # n_display번째 epoch 마다 출력
        # RMSE with training data
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})        
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_List.append(train_error) # train error 기록

        # RMSE with test data
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_List.append(test_error) # test error 기록

def priceOfTommorow():  # 내일 종가를 예측하는 함수   
    recent_data = np.array(
        [ 
            x[len(x)-seq_length : ] 
        ]
    ) # 최신 데이터를 sequence 길이만큼 slicing
    
    test_predict = sess.run(
        hypothesis, 
        feed_dict={ X: recent_data }
    ) # feed recent data
    
    test_predict = reverse_minMaxScaler( price, test_predict )
    string = "내일의 종가는 : "+ str(int(test_predict[0]))
    return string

class PypltQt(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.setLayout(self.layout)
        self.setGeometry(200, 200, 1200, 400)

    def initUI(self):        
        self.setWindowTitle("딥러닝 주가 예측")
        
        # canvas Layout        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        canvasLayout = QVBoxLayout()
        canvasLayout.addWidget(self.canvas)
#         canvasLayout.addStretch(1)
        
        # canvas Layout2
        self.fig2 = plt.Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        canvasLayout2 = QVBoxLayout()
        canvasLayout2.addWidget(self.canvas2)
#         canvasLayout2.addStretch(1)        
        
        # button layout
        self.pushButton = QPushButton("DRAW Graph")
        self.pushButton.clicked.connect(self.btnClicked)
        btnLayout = QVBoxLayout()
        btnLayout.addWidget(self.pushButton)
        
        # label layout
        self.labelA =QLabel(self)
        self.labelA.setAlignment(Qt.AlignCenter)
        labelLayout = QVBoxLayout()
        labelLayout.addWidget(self.labelA)
        
        # All layout is in the Horizontal Box  
        box_v = QVBoxLayout()
        box_v.addLayout(btnLayout) 
        box_v.addLayout(labelLayout)    
        
        box_h = QHBoxLayout()
        box_h.addLayout(canvasLayout)
        box_h.addLayout(canvasLayout2)    
        box_h.addLayout(box_v)
        self.layout = box_h
        
    def btnClicked(self):
        string = priceOfTommorow()
        self.labelA.setText(string)
        
        ax = self.fig.add_subplot(111)
        ax.set_title('error')
        ax.plot(train_error_List, 'b', label="train error")
        ax.plot(test_error_List, 'r', label="test error")
        ax.legend()
        self.canvas.draw()
        
        ax2 = self.fig2.add_subplot(111)
        ax2.set_title('prediction')
        ax2.plot(testY, 'b', label="target label")
        ax2.plot(test_predict, 'r', label="prediction")
        ax2.legend()
        self.canvas2.draw()
        
if __name__ == "__main__":
    qapp = QApplication(sys.argv)
    app = PypltQt()
    app.show()
    qapp.exec_()