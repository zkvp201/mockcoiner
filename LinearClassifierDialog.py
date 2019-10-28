#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import pybithumb
import math
from pybithumb import Bithumb

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import time

# 시가총액 top 5 코인 리스트
stockList = [
    "BTC", 
    "ETH", # 이더리움
    "XRP", # 리플
    "BCH", # 비트코인 캐시
    "LTC" # 라이트코인
]

# # 포트폴리오(비율) 합 = 1
portfolio_rate = {
    "BTC" : [0.1],
    "ETH" : [0.1], # 이더리움
    "XRP" : [0.1], # 리플
    "BCH" : [0.1], # 비트코인 캐시
    "LTC" : [0.1], # 라이트코인
    "cash" : [0.5] # 현금 100억
}
cash_all = 1e+7 # 전재산 천만원

df_portfolio = pd.DataFrame()

for stock in stockList:
    df_portfolio[stock] = pybithumb.get_ohlcv(stock)["close"][-300:]

portfolio_coin = {}
df_portfolio_rate = pd.DataFrame.from_dict(portfolio_rate) 

for stock in df_portfolio_rate:
    investment = float(df_portfolio_rate[stock]) * cash_all
    
    if(stock == "cash"):
        n_coin = investment
    else:
        coin_price = df_portfolio[stock][6]
        n_coin = investment / coin_price
    portfolio_coin[stock] = n_coin

# Dialog 코드
ui_file_path = "Dialog2.ui"
form_class = uic.loadUiType( ui_file_path )[0]

class LogInDialog(QDialog , form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.function) # 시그널과 슬롯 연결

    def function(self):
        self.pushButton.setText("머신러닝 중...")
        self.printText()
        
    def printText(self):        
        # 출력부분
        df_7ma = pd.DataFrame()
        for stock in stockList:
            df_7ma[stock] = df_portfolio[stock].rolling(window=7).mean()

        df_7ma = df_7ma.dropna()

        def rebalancedRate(rebalanced_rate, stock, stock_mean, stock_price):
            n_stock = portfolio_coin[stock]
            balance_rate = stock_mean / stock_price # 조정 비율: 낮을 수록 시장 호황
            change = 1-balance_rate # 조정비율이 낮으면 +
            balance_rate +=  change/10

            if (balance_rate < 1): # 시장 과열시 매도
                if(n_stock > 0): # 주식을 매도할 경우 보유주가 있는지 확인해야 한다.            
                    rebalanced_rate = float(df_portfolio_rate[stock]) * balance_rate 
                else:
                    print("보유주 없음")
                    
            elif(balance_rate == 1):
                pass

            else: # 시장 침체시 매수
                rebalanced_rate = float(df_portfolio_rate[stock]) * balance_rate #+ resistance

            return rebalanced_rate

        df_rebalancing_asset = df_portfolio.iloc[:].copy()
        df_rebalancing_asset[:] = np.NaN
        df_rebalancing_asset["cash"] = np.NaN
        df_rebalancing_asset["assets"] = np.NaN
        df_rebalancing_asset.index.name = "datetime"

        def updateRatio( df_rebalanced_rate ):
            stock_ratio = 0 # 총 투자비율    
            for stock in stockList:
                stock_ratio += float(df_portfolio_rate[stock])
                        
            if(stock_ratio > 1):        
                for stock in stockList:
                    df_portfolio_rate[stock] = float(df_rebalanced_rate[stock]) / stock_ratio # 투자 비율 업데이트
                
                stock_ratio = 1
            else:
                for stock in stockList:
                    df_portfolio_rate[stock] = float(df_rebalanced_rate[stock]) # 투자 비율 업데이트
                
            df_portfolio_rate["cash"][0] =  1 - stock_ratio # 현금 비율 = 1 - 총 투자비율

        i = 6 # 해당 일에는 비율을 변경하지 않음
        string = "i = "+ str(i)
        self.textEdit.append( string )

        assets = 0
        for stock in stockList:
            stock_price = df_portfolio[stock]
            assets += portfolio_coin[stock] * stock_price[i]

        assets += portfolio_coin["cash"]
        string = "현재 자산가치: "+ str(assets)
        self.textEdit.append( string )
        df_rebalancing_asset["assets"][i] = assets 

        string = "포트폴리오(비율): \n"+ str(df_portfolio_rate)
        self.textEdit.append( string ) 

        # 리밸런싱 로그에 투자 비율 기록
        for stock in df_portfolio_rate:      
            tmp = df_portfolio_rate[stock]
            df_rebalancing_asset[stock][i] = tmp

        self.textEdit.append( '='*32 )

        freq_rebalance = 1 # rebalancing 주기    
        for i in range(7, 299, freq_rebalance): 
            string = "i = "+ str(i)
            self.textEdit.append( string )
            
            assets = 0    
            for stock in stockList:
                stock_price = df_portfolio[stock]
                assets += portfolio_coin[stock] * stock_price[i]
            
            assets += portfolio_coin["cash"]
            
            string = "현재 자산가치: "+ str(assets)
            self.textEdit.append( string )

            df_rebalancing_asset["assets"][i] = assets 
            
            df_rebalanced_rate = pd.DataFrame.from_dict(df_portfolio_rate)
            for stock in stockList:
                stock_mean = df_7ma[stock][i-8] # 전날 7일 평균
                stock_price = df_portfolio[stock][i-1] # 전날 종가
                df_rebalanced_rate[stock] = rebalancedRate( df_rebalanced_rate[stock], stock, stock_mean, stock_price )
            updateRatio(df_rebalanced_rate) # 현금비율 조정

            string = "포트폴리오(바꿀비율): \n"+ str(df_portfolio_rate)
            self.textEdit.append( string )
            
            # 현금 조정
            rebalanced_cash = assets
            for stock in stockList:
                stock_price = df_portfolio[stock][i]
                investment_stock = assets * float(df_portfolio_rate[stock][0]) # 한 종목당 투자금
                portfolio_coin[stock] = investment_stock / stock_price # 보유 코인 수 계산
                rebalanced_cash -= investment_stock
                
            portfolio_coin["cash"] = rebalanced_cash
            
            for stock in df_portfolio_rate:      
                tmp = df_portfolio_rate[stock]
                df_rebalancing_asset[stock][i] = tmp
                
            self.textEdit.append( '='*32 )

        df_rebalancing_asset["profit"] = df_rebalancing_asset["assets"] - df_rebalancing_asset["assets"].shift(1)

        df_rebalancing_asset["classification"] = np.where( 
            df_rebalancing_asset["profit"] > 0,
            1, 0 )

        df_rebalancing_asset = df_rebalancing_asset.dropna()

        # 머신러닝 코드
        tf.set_random_seed(777)  # for reproducibility

        x_data = df_rebalancing_asset.iloc[:, :5] # 293 x 5 column
        y_data = df_rebalancing_asset.iloc[:, [-1]]  # 293 x 1

        X = tf.placeholder(tf.float32, shape=[None, 5])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([5, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))

        train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

        predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

        result = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            feed = {X: x_data, Y: y_data}
            epochs = 10**4 + 1
            for step in range(epochs):
                sess.run(train, feed_dict=feed)
                    
            h, c, result = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)

        # print("머신러닝 분석 결과 예측 정확도:")
        string = "머신러닝 분석 결과 예측 정확도: "+ str(result)
        self.textEdit.append( string )

        i = 299
        # 비율 조정
        df_rebalanced_rate = pd.DataFrame.from_dict(df_portfolio_rate)
        for stock in stockList:
            stock_mean = df_7ma[stock][i-8] # 전날 7일 평균
            stock_price = df_portfolio[stock][i-1] # 전날 종가
            df_rebalanced_rate[stock] = rebalancedRate( df_rebalanced_rate[stock], stock, stock_mean, stock_price)       
        updateRatio(df_rebalanced_rate) # 현금비율 조정

        # 현금 조정
        rebalanced_cash = assets
        for stock in stockList:
            stock_price = df_portfolio[stock][i]
            investment_stock = assets * float(df_portfolio_rate[stock][0]) # 한 종목당 투자금
            portfolio_coin[stock] = investment_stock / stock_price # 보유 코인 수 계산
            rebalanced_cash -= investment_stock

        portfolio_coin["cash"] = rebalanced_cash

        string = "\n오늘의 추천 비율: \n"+ str(df_portfolio_rate)
        self.textEdit.append( string )

class MyWindow(QWidget):
    def __init__(self):
        super().__init__() #메인윈도우 셋업
        self.setupUI()

    def setupUI(self):
        self.setGeometry(800, 200, 300, 300)
        self.setWindowTitle("PyStock v0.1")

        self.pushButton = QPushButton("Sign In")
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.pushButton)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def pushButtonClicked(self): #버튼이 눌리면 다이얼로그가 생성됨
        dlg = LogInDialog()
        dlg.exec_()

# if __name__ == "__main__":
app = QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()