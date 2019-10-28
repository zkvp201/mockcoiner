# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import pybithumb
from PyQt5.QtGui import *
import ror
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
import Real_trade
import pymysql
import datetime
import Real_trade as rt
import pandas as pd
import operator
import tensorflow as tf

QSS ='''
QToolTip
{
     border: 1px solid black;
     background-color: #ffa02f;
     padding: 1px;
     border-radius: 3px;
     opacity: 100;
}

QWidget
{
    color: #b1b1b1;
    background-color: #323232;
}

QWidget:item:hover
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #ca0619);
    color: #000000;
}

QWidget:item:selected
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}

QMenuBar::item
{
    background: transparent;
}

QMenuBar::item:selected
{
    background: transparent;
    border: 1px solid #ffaa00;
}

QMenuBar::item:pressed
{
    background: #444;
    border: 1px solid #000;
    background-color: QLinearGradient(
        x1:0, y1:0,
        x2:0, y2:1,
        stop:1 #212121,
        stop:0.4 #343434/*,
        stop:0.2 #343434,
        stop:0.1 #ffaa00*/
    );
    margin-bottom:-1px;
    padding-bottom:1px;
}

QMenu
{
    border: 1px solid #000;
}

QMenu::item
{
    padding: 2px 20px 2px 20px;
}

QMenu::item:selected
{
    color: #000000;
}

QWidget:disabled
{
    color: #404040;
    background-color: #323232;
}

QAbstractItemView
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0.1 #646464, stop: 1 #5d5d5d);
}

QWidget:focus
{
    /*border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);*/
}

QLineEdit
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0 #646464, stop: 1 #5d5d5d);
    padding: 1px;
    border-style: solid;
    border: 1px solid #1e1e1e;
    border-radius: 5;
}

QPushButton
{
    color: #b1b1b1;
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);
    border-width: 1px;
    border-color: #1e1e1e;
    border-style: solid;
    border-radius: 6;
    padding: 3px;
    font-size: 12px;
    padding-left: 5px;
    padding-right: 5px;
}

QPushButton:pressed
{
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);
}

QComboBox
{
    selection-background-color: #ffaa00;
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);
    border-style: solid;
    border: 1px solid #1e1e1e;
    border-radius: 5;
}

QComboBox:hover,QPushButton:hover
{
    border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}


QComboBox:on
{
    padding-top: 3px;
    padding-left: 4px;
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);
    selection-background-color: #ffaa00;
}

QComboBox QAbstractItemView
{
    border: 2px solid darkgray;
    selection-background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}

QComboBox::drop-down
{
     subcontrol-origin: padding;
     subcontrol-position: top right;
     width: 15px;

     border-left-width: 0px;
     border-left-color: darkgray;
     border-left-style: solid; /* just a single line */
     border-top-right-radius: 3px; /* same radius as the QComboBox */
     border-bottom-right-radius: 3px;
 }

QComboBox::down-arrow
{
     image: url(:/down_arrow.png);
}

QGroupBox:focus
{
border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}

QTextEdit:focus
{
    border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
}

QScrollBar:horizontal {
     border: 1px solid #222222;
     background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);
     height: 7px;
     margin: 0px 16px 0 16px;
}

QScrollBar::handle:horizontal
{
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 0.5 #d7801a, stop: 1 #ffa02f);
      min-height: 20px;
      border-radius: 2px;
}

QScrollBar::add-line:horizontal {
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 1 #d7801a);
      width: 14px;
      subcontrol-position: right;
      subcontrol-origin: margin;
}

QScrollBar::sub-line:horizontal {
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 1 #d7801a);
      width: 14px;
     subcontrol-position: left;
     subcontrol-origin: margin;
}

QScrollBar::right-arrow:horizontal, QScrollBar::left-arrow:horizontal
{
      border: 1px solid black;
      width: 1px;
      height: 1px;
      background: white;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal
{
      background: none;
}

QScrollBar:vertical
{
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);
      width: 7px;
      margin: 16px 0 16px 0;
      border: 1px solid #222222;
}

QScrollBar::handle:vertical
{
      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 0.5 #d7801a, stop: 1 #ffa02f);
      min-height: 20px;
      border-radius: 2px;
}

QScrollBar::add-line:vertical
{
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);
      height: 14px;
      subcontrol-position: bottom;
      subcontrol-origin: margin;
}

QScrollBar::sub-line:vertical
{
      border: 1px solid #1b1b19;
      border-radius: 2px;
      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #d7801a, stop: 1 #ffa02f);
      height: 14px;
      subcontrol-position: top;
      subcontrol-origin: margin;
}

QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical
{
      border: 1px solid black;
      width: 1px;
      height: 1px;
      background: white;
}


QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical
{
      background: none;
}

QTextEdit
{
    background-color: #242424;
}

QPlainTextEdit
{
    background-color: #242424;
}

QHeaderView::section
{
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #616161, stop: 0.5 #505050, stop: 0.6 #434343, stop:1 #656565);
    color: white;
    padding-left: 4px;
    border: 1px solid #6c6c6c;
}

QCheckBox:disabled
{
color: #414141;
}

QDockWidget::title
{
    text-align: center;
    spacing: 3px; /* spacing between items in the tool bar */
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop: 0.5 #242424, stop:1 #323232);
}

QDockWidget::close-button, QDockWidget::float-button
{
    text-align: center;
    spacing: 1px; /* spacing between items in the tool bar */
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop: 0.5 #242424, stop:1 #323232);
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover
{
    background: #242424;
}

QDockWidget::close-button:pressed, QDockWidget::float-button:pressed
{
    padding: 1px -1px -1px 1px;
}

QMainWindow::separator
{
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop: 0.5 #151515, stop: 0.6 #212121, stop:1 #343434);
    color: white;
    padding-left: 4px;
    border: 1px solid #4c4c4c;
    spacing: 3px; /* spacing between items in the tool bar */
}

QMainWindow::separator:hover
{

    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d7801a, stop:0.5 #b56c17 stop:1 #ffa02f);
    color: white;
    padding-left: 4px;
    border: 1px solid #6c6c6c;
    spacing: 3px; /* spacing between items in the tool bar */
}

QToolBar::handle
{
     spacing: 3px; /* spacing between items in the tool bar */
     background: url(:/images/handle.png);
}

QMenu::separator
{
    height: 2px;
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop: 0.5 #151515, stop: 0.6 #212121, stop:1 #343434);
    color: white;
    padding-left: 4px;
    margin-left: 10px;
    margin-right: 5px;
}

QProgressBar
{
    border: 2px solid grey;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk
{
    background-color: #d7801a;
    width: 2.15px;
    margin: 0.5px;
}

QTabBar::tab {
    color: #b1b1b1;
    border: 1px solid #444;
    border-bottom-style: none;
    background-color: #323232;
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 3px;
    padding-bottom: 2px;
    margin-right: -1px;
}

QTabWidget::pane {
    border: 1px solid #444;
    top: 1px;
}

QTabBar::tab:last
{
    margin-right: 0; /* the last selected tab has nothing to overlap with on the right */
    border-top-right-radius: 3px;
}

QTabBar::tab:first:!selected
{
 margin-left: 0px; /* the last selected tab has nothing to overlap with on the right */


    border-top-left-radius: 3px;
}

QTabBar::tab:!selected
{
    color: #b1b1b1;
    border-bottom-style: solid;
    margin-top: 3px;
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:.4 #343434);
}

QTabBar::tab:selected
{
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    margin-bottom: 0px;
}

QTabBar::tab:!selected:hover
{
    /*border-top: 2px solid #ffaa00;
    padding-bottom: 3px;*/
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:0.4 #343434, stop:0.2 #343434, stop:0.1 #ffaa00);
}

QRadioButton::indicator:checked, QRadioButton::indicator:unchecked{
    color: #b1b1b1;
    background-color: #323232;
    border: 1px solid #b1b1b1;
    border-radius: 6px;
}

QRadioButton::indicator:checked
{
    background-color: qradialgradient(
        cx: 0.5, cy: 0.5,
        fx: 0.5, fy: 0.5,
        radius: 1.0,
        stop: 0.25 #ffaa00,
        stop: 0.3 #323232
    );
}

QCheckBox::indicator{
    color: #b1b1b1;
    background-color: #323232;
    border: 1px solid #b1b1b1;
    width: 9px;
    height: 9px;
}

QRadioButton::indicator
{
    border-radius: 6px;
}

QRadioButton::indicator:hover, QCheckBox::indicator:hover
{
    border: 1px solid #ffaa00;
}

QCheckBox::indicator:checked
{
    image:url(:/images/checkbox.png);
}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled
{
    border: 1px solid #444;
}
'''
form_class = uic.loadUiType("Coin_GUI.ui")[0]
ML_class = uic.loadUiType("Dialog2.ui")[0]

target = []
price = []
ma5 = []
noise = []
noise_df = []
_id_ =""
tickers = ["BTC", "ETH", "BCH", "LTC","ETC"]
ticker_list = ["BTC", "ETH", "BCH", "LTC","ETC"]
stockList = ["BTC","ETH","BCH","LTC","ETC"]

# # 포트폴리오(비율) 합 = 1
portfolio_rate = {
    tickers[0]: [0.1],
    tickers[1]: [0.1],  
    tickers[2]: [0.1],  
    tickers[3]: [0.1],  
    tickers[4]: [0.1],  
    "cash": [0.5]  
}
cash_all = 1e+7  # 전재산 천만원
df_portfolio = pd.DataFrame()
portfolio_coin = {}
df_portfolio_rate = pd.DataFrame.from_dict(portfolio_rate)
# Dialog 코드
ui_file_path = "Dialog2.ui"
ML_class = uic.loadUiType(ui_file_path)[0]
###############################
class MLDialog(QDialog, ML_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.function) 

    def function(self):
        self.pushButton.setText("머신러닝 중...")
        self.printText()

    def printText(self):
        # 출력부분
        df_portfolio_rate = pd.DataFrame.from_dict(portfolio_rate)

        for stock in stockList:
            df_portfolio[stock] = pybithumb.get_ohlcv(stock)["close"][-300:]

        for stock in df_portfolio_rate:
            investment = float(df_portfolio_rate[stock]) * cash_all

            if (stock == "cash"):
                n_coin = investment
            else:
                coin_price = df_portfolio[stock][6]
                n_coin = investment / coin_price
            portfolio_coin[stock] = n_coin

        df_7ma = pd.DataFrame()
        for stock in stockList:
            df_7ma[stock] = df_portfolio[stock].rolling(window=7).mean()

        df_7ma = df_7ma.dropna()

        def rebalancedRate(rebalanced_rate, stock, stock_mean, stock_price):
            n_stock = portfolio_coin[stock]
            balance_rate = stock_mean / stock_price  # 조정 비율: 낮을 수록 시장 호황
            change = 1 - balance_rate  # 조정비율이 낮으면 +
            balance_rate += change / 10

            if (balance_rate < 1):  # 시장 과열시 매도
                if (n_stock > 0):  # 주식을 매도할 경우 보유주가 있는지 확인해야 한다.
                    rebalanced_rate = float(df_portfolio_rate[stock]) * balance_rate
                else:
                    print("보유주 없음")

            elif (balance_rate == 1):
                pass

            else:  # 시장 침체시 매수
                rebalanced_rate = float(df_portfolio_rate[stock]) * balance_rate  # + resistance

            return rebalanced_rate

        df_rebalancing_asset = df_portfolio.iloc[:].copy()
        df_rebalancing_asset[:] = np.NaN
        df_rebalancing_asset["cash"] = np.NaN
        df_rebalancing_asset["assets"] = np.NaN
        df_rebalancing_asset.index.name = "datetime"

        def updateRatio(df_rebalanced_rate):
            stock_ratio = 0  # 총 투자비율
            for stock in stockList:
                stock_ratio += float(df_portfolio_rate[stock])

            if (stock_ratio > 1):
                for stock in stockList:
                    df_portfolio_rate[stock] = float(df_rebalanced_rate[stock]) / stock_ratio  # 투자 비율 업데이트

                stock_ratio = 1
            else:
                for stock in stockList:
                    df_portfolio_rate[stock] = float(df_rebalanced_rate[stock])  # 투자 비율 업데이트

            df_portfolio_rate["cash"][0] = 1 - stock_ratio  # 현금 비율 = 1 - 총 투자비율

        i = 6  # 해당 일에는 비율을 변경하지 않음
        string = "i = " + str(i)
        self.textEdit.append(string)

        assets = 0
        for stock in stockList:
            stock_price = df_portfolio[stock]
            assets += portfolio_coin[stock] * stock_price[i]

        assets += portfolio_coin["cash"]
        string = "현재 자산가치: " + str(assets)
        self.textEdit.append(string)
        df_rebalancing_asset["assets"][i] = assets

        string = "포트폴리오(비율): \n" + str(df_portfolio_rate)
        self.textEdit.append(string)

        # 리밸런싱 로그에 투자 비율 기록
        for stock in df_portfolio_rate:
            tmp = df_portfolio_rate[stock]
            df_rebalancing_asset[stock][i] = tmp

        self.textEdit.append('=' * 32)

        freq_rebalance = 1  # rebalancing 주기
        for i in range(7, 299, freq_rebalance):
            string = "i = " + str(i)
            self.textEdit.append(string)

            assets = 0
            for stock in stockList:
                stock_price = df_portfolio[stock]
                assets += portfolio_coin[stock] * stock_price[i]

            assets += portfolio_coin["cash"]

            string = "현재 자산가치: " + str(assets)
            self.textEdit.append(string)

            df_rebalancing_asset["assets"][i] = assets

            df_rebalanced_rate = pd.DataFrame.from_dict(df_portfolio_rate)
            for stock in stockList:
                stock_mean = df_7ma[stock][i - 8]  # 전날 7일 평균
                stock_price = df_portfolio[stock][i - 1]  # 전날 종가
                df_rebalanced_rate[stock] = rebalancedRate(df_rebalanced_rate[stock], stock, stock_mean, stock_price)
            updateRatio(df_rebalanced_rate)  # 현금비율 조정

            string = "포트폴리오(바꿀비율): \n" + str(df_portfolio_rate)
            self.textEdit.append(string)

            # 현금 조정
            rebalanced_cash = assets
            for stock in stockList:
                stock_price = df_portfolio[stock][i]
                investment_stock = assets * float(df_portfolio_rate[stock][0])  # 한 종목당 투자금
                portfolio_coin[stock] = investment_stock / stock_price  # 보유 코인 수 계산
                rebalanced_cash -= investment_stock

            portfolio_coin["cash"] = rebalanced_cash

            for stock in df_portfolio_rate:
                tmp = df_portfolio_rate[stock]
                df_rebalancing_asset[stock][i] = tmp

            self.textEdit.append('=' * 32)

        df_rebalancing_asset["profit"] = df_rebalancing_asset["assets"] - df_rebalancing_asset["assets"].shift(1)

        df_rebalancing_asset["classification"] = np.where(
            df_rebalancing_asset["profit"] > 0,
            1, 0)

        df_rebalancing_asset = df_rebalancing_asset.dropna()

        # 머신러닝 코드
        tf.set_random_seed(777)  # for reproducibility

        x_data = df_rebalancing_asset.iloc[:, :5]  # 293 x 5 column
        y_data = df_rebalancing_asset.iloc[:, [-1]]  # 293 x 1

        X = tf.placeholder(tf.float32, shape=[None, 5])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        W = tf.Variable(tf.random_normal([5, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

        train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

        predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

        result = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            feed = {X: x_data, Y: y_data}
            epochs = 10 ** 4 + 1
            for step in range(epochs):
                sess.run(train, feed_dict=feed)

            h, c, result = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)

        # print("머신러닝 분석 결과 예측 정확도:")
        string = "머신러닝 분석 결과 예측 정확도: " + str(result)
        self.textEdit.append(string)

        i = 299
        # 비율 조정
        df_rebalanced_rate = pd.DataFrame.from_dict(df_portfolio_rate)
        for stock in stockList:
            stock_mean = df_7ma[stock][i - 8]  # 전날 7일 평균
            stock_price = df_portfolio[stock][i - 1]  # 전날 종가
            df_rebalanced_rate[stock] = rebalancedRate(df_rebalanced_rate[stock], stock, stock_mean, stock_price)
        updateRatio(df_rebalanced_rate)  # 현금비율 조정

        # 현금 조정
        rebalanced_cash = assets
        for stock in stockList:
            stock_price = df_portfolio[stock][i]
            investment_stock = assets * float(df_portfolio_rate[stock][0])  # 한 종목당 투자금
            portfolio_coin[stock] = investment_stock / stock_price  # 보유 코인 수 계산
            rebalanced_cash -= investment_stock

        portfolio_coin["cash"] = rebalanced_cash

        string = "\n오늘의 추천 비율: \n" + str(df_portfolio_rate)
        self.textEdit.append(string)
###############################################################################
class TWorker(QThread):
    finished = pyqtSignal()

    def run(self):
        while True:
            self.realtrade(tickers)
            self.finished.emit() #작업이 다 끝나면 신호를 발생시킴, pyqtsignal에 emit이라는 함수가 정의 되어 있음(시그널 발생)
            self.msleep(1500)


    def realtrade(self,ticker_list):
        now = datetime.datetime.now()
        time1, time2 = rt.make_times(now)

        for i in range(5):
            target_temp, noise_temp = rt.cal_target(ticker_list[i])
            target.append(target_temp)  # 목표 가격을 가져온다.
            noise.append(noise_temp)  # 노이즈를 생성

            temp = pybithumb.get_current_price(ticker_list[i])
            price.append(temp)  # 현재 가격을 가져온다.

            temp = rt.cal_moving_average(ticker_list[i], window=5)
            ma5.append(temp)  # 5일 이동평균선을 가져온다

        name = ["coin1", "coin2", "coin3", "coin4", "coin5"]
        noise_list = dict(zip(name, noise))
        ordered_dict = sorted(noise_list.items(), key=operator.itemgetter(1))
        ordered_dict = ordered_dict[:3]

        tempList = []
        for coin_tuple in ordered_dict:
            tempList.append(coin_tuple[0])
        noise_list = tempList
        #
        df = pd.DataFrame(noise_list)
        df = df.T
        df = df.rename(columns={0: 'TOP1', 1: 'TOP2', 2: 'TOP3'})
        df['TOP1'] = noise_list[0]
        df['TOP2'] = noise_list[1]
        df['TOP3'] = noise_list[2]
        #
        # # 필터링 조건 활성화
        df['coin_1'] = ((df['TOP1'] == 'coin1') | (df['TOP2'] == 'coin1') | (df['TOP3'] == 'coin1'))
        df['coin_2'] = ((df['TOP1'] == 'coin2') | (df['TOP2'] == 'coin2') | (df['TOP3'] == 'coin2'))
        df['coin_3'] = ((df['TOP1'] == 'coin3') | (df['TOP2'] == 'coin3') | (df['TOP3'] == 'coin3'))
        df['coin_4'] = ((df['TOP1'] == 'coin4') | (df['TOP2'] == 'coin4') | (df['TOP3'] == 'coin4'))
        df['coin_5'] = ((df['TOP1'] == 'coin5') | (df['TOP2'] == 'coin5') | (df['TOP3'] == 'coin5'))
        #
        df = df.T
        for i in range(5):
            a=df.iloc[i]
            noise_df.append(a)

class CreateDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setupUI()

        self.id = None
        self.password = None

    def setupUI(self):
        self.setGeometry(1100, 200, 300, 100)
        self.setWindowTitle("Sign In")
        self.setWindowIcon(QIcon('icon.png'))

        label1 = QLabel("ID: ")
        label2 = QLabel("Password: ")

        self.lineEdit1 = QLineEdit()
        self.lineEdit2 = QLineEdit()
        self.pushButton1= QPushButton("Sign In")
        self.pushButton1.clicked.connect(self.pushButtonClicked)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.lineEdit1, 0, 1)
        layout.addWidget(self.pushButton1, 0, 2)
        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.lineEdit2, 1, 1)

        self.setLayout(layout)

    def pushButtonClicked(self):
        self.id = self.lineEdit1.text()
        self.password = self.lineEdit2.text()

        def getConnection():
            conn = pymysql.connect(
                host='127.0.0.1',
                user='root',
                password='sinbu753951!',
                db='mockcoiner_db',
                charset='utf8'
            )
            return conn

        conn = pymysql.connect(host='db-2pm0u.pub-cdb.ntruss.com',port=3306 ,user='mockcoiner_dba',password='sinbu753951!',db='mockcoiner_db',charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)

        input_email = self.id

        input_pwd = self.password

        sql = """
        INSERT INTO mockcoiner_db.user (email, password, signup_date, balance) VALUES ('%s', '%s', '%s', '100000000');
        """ % (input_email, input_pwd, datetime.datetime.now())

        curs.execute(sql)
        conn.commit()
        conn.close()

        self.close()

class LogInDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setupUI()

        self.id = None
        self.password = None

    def setupUI(self): #다이얼로그 화면을 셋업 하는거임
        self.setGeometry(1100, 200, 300, 100)
        self.setWindowTitle("Log In")
        self.setWindowIcon(QIcon('icon.png'))

        label1 = QLabel("ID: ")
        label2 = QLabel("Password: ")

        self.lineEdit1 = QLineEdit()
        self.lineEdit2 = QLineEdit()
        self.pushButton1= QPushButton("Log In")
        self.pushButton1.clicked.connect(self.pushButtonClicked)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.lineEdit1, 0, 1)
        layout.addWidget(self.pushButton1, 0, 2)
        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.lineEdit2, 1, 1)

        self.setLayout(layout)


    def pushButtonClicked(self):
        self.id = self.lineEdit1.text()
        self.password = self.lineEdit2.text()
        self.close()

class Worker(QThread):
    finished = pyqtSignal(dict)

    def run(self):
        while True:
            data = {}

            for ticker in tickers:
                data[ticker] = self.get_market_infos(ticker)

            self.finished.emit(data) #작업이 다 끝나면 신호를 발생시킴, pyqtsignal에 emit이라는 함수가 정의 되어 있음(시그널 발생)
            self.msleep(500)

    def get_market_infos(self, ticker):
        try:
            df = pybithumb.get_ohlcv(ticker)
            ma5 = df['close'].rolling(window=5).mean()

            price = pybithumb.get_current_price(ticker)
            last_ma5 = ma5[-2]

            state = None
            if price > last_ma5:
                state = "상승장"
            else:
                state = "하락장"

            return (price, last_ma5, state)
        except:
            return (None, None, None)
class BacktestingWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.setLayout(self.layout)
        self.setGeometry(200, 200, 800, 400)

    def initUI(self):

        self.pushButton = QPushButton("DRAW Graph")
        self.lineEdit = QLineEdit("수익률 : ")
        self.lineEdit2 = QLineEdit("최대손실율 : ")
        self.pushButton.clicked.connect(self.btnClicked)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        # btn layout
        btnLayout = QVBoxLayout()
        btnLayout.addWidget(self.canvas)

        # canvas Layout
        canvasLayout = QVBoxLayout()
        canvasLayout.addWidget(self.pushButton)
        canvasLayout.addWidget(self.lineEdit)
        canvasLayout.addWidget(self.lineEdit2)
        canvasLayout.addStretch(1)

        self.layout = QHBoxLayout()
        self.layout.addLayout(btnLayout)
        self.layout.addLayout(canvasLayout)

    def btnClicked(self):
        

        df = ror.realtime_noise_ror(tickers[0])
        ax = self.fig.add_subplot(311)
        ax.set_title('Coin Price', size=10)
        ax.plot(df['close'])

        ax = self.fig.add_subplot(313)
        ax.plot(df['cumprod2'], 'r')
        ax.set_title('ROR', size=10)
        mdd = df['mdd2'].max()
        mdd = round(mdd,4)
        ror2 = df['cumprod2'][-2]
        ror2 = round(ror2,4)
        self.lineEdit.setText("최대손실율 : "+str(mdd))
        self.lineEdit2.setText("수익률 : "+ str(ror2))

        self.canvas.draw()
class BacktestingWindow_dist(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.setLayout(self.layout)
        self.setGeometry(200, 200, 800, 400)

    def initUI(self):

        self.pushButton = QPushButton("DRAW Graph")
        self.lineEdit = QLineEdit(" TOP3 수익률 : ")
        self.lineEdit2 = QLineEdit("TOP3 최대손실율 : ")
        self.lineEdit3 = QLineEdit(" TOP5 수익률 : ")
        self.lineEdit4 = QLineEdit("TOP5 최대손실율 : ")
        self.pushButton.clicked.connect(self.btnClicked)


        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        # btn layout
        btnLayout = QVBoxLayout()
        btnLayout.addWidget(self.canvas)

        # canvas Layout
        canvasLayout = QVBoxLayout()
        canvasLayout.addWidget(self.pushButton)
        canvasLayout.addWidget(self.lineEdit)
        canvasLayout.addWidget(self.lineEdit2)
        canvasLayout.addWidget(self.lineEdit3)
        canvasLayout.addWidget(self.lineEdit4)
        canvasLayout.addStretch(1)



        self.layout = QHBoxLayout()
        self.layout.addLayout(btnLayout)
        self.layout.addLayout(canvasLayout)

    def btnClicked(self):

        df = ror.Top3_dist_ror(tickers)
        ax = self.fig.add_subplot(111)
        ax.plot(df['cumprod2'], label=tickers[0])
        ax.plot(df['coin2'], label=tickers[1])
        ax.plot(df['coin3'], label=tickers[2])
        ax.plot(df['coin4'], label=tickers[3])
        ax.plot(df['coin5'], label=tickers[4])
        ax.plot(df['top3_ror'], label= 'top3_ror')
        ax.plot(df['top5_ror'], label='top5_ror')
        ax.legend()
        ax.set_title('Coin Backtesting')

        top3_mdd = df['top3_MDD'].max()
        top3_mdd = round(top3_mdd,4)
        top5_mdd = df['top5_MDD'].max()
        top5_mdd = round(top5_mdd,4)
        top3_ror = df['top3_ror'][-2]
        top3_ror = round(top3_ror,4)
        top5_ror = df['top5_ror'][-2]
        top5_ror = round(top5_ror,4)

        self.lineEdit.setText("TOP3 MDD : "+str(top3_mdd))
        self.lineEdit2.setText("TOP3 ROR : "+ str(top3_ror))
        self.lineEdit3.setText("TOP5 MDD : "+str(top5_mdd))
        self.lineEdit4.setText("TOP5 ROR : "+ str(top5_ror))
        self.canvas.draw()

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        #UI 초기 설정
        self.setupUi(self)
        self.setWindowTitle("Mock Coiner ver.0.1")
        self.setWindowIcon(QIcon("icon.png"))
        self.actionCreateAccount.triggered.connect(self.createaccount)
        self.actionExit.setShortcut('Ctrl+Q')
        self.actionExit.triggered.connect(QCoreApplication.instance().quit)

        self.tradeBtn.setDisabled(True)
        self.closeBtn.setDisabled(True)
        self.comboBox.setDisabled(True)
        self.comboBox2.setDisabled(True)
        self.comboBox3.setDisabled(True)
        self.comboBox4.setDisabled(True)
        self.comboBox5.setDisabled(True)

        self.inputCoin.setDisabled(True)
        self.resetCoin.setDisabled(True)
        self.sellCoin.setDisabled(True)

        self.pushButton_2.setDisabled(True)
        self.pushButton_3.setDisabled(True)

        self.pushButton_5.setDisabled(True)
        #UI 초기 설정

        #Thread Event
        self.tableWidget.setRowCount(len(tickers))
        self.worker = Worker()
        self.worker.finished.connect(self.update_table_widget) # if thread works finished, send emit signal to slot(update_table_widget function)
        self.worker.start() #start can do run function

        #Event
        self.pushButton.clicked.connect(QCoreApplication.instance().quit) # pushButton ==  quit button
        self.pushButton_2.clicked.connect(self.backTestingModule) # pushButton_2 == backtesting Moudule button
        self.pushButton_3.clicked.connect(self.backTestingMoudule_dist) #pushButton 3 == Dist Backtesting Module
        self.pushButton_5.clicked.connect(self.MLTestingModule) # pushButton_5 == ML Testing Module
        self.loginbtn.clicked.connect(self.loginevent) #Login Btn
        
        #Thread Event2
        self.tworker = TWorker() # real trade thread
        self.tworker.finished.connect(self.try_buy) #real trade thread slot

        self.tradeBtn.clicked.connect(self.startThread) # real trade start btn
        self.closeBtn.clicked.connect(self.resetThread)# real trade thread terminate btn
        self.updateBtn.clicked.connect(self.updateCoin)


        self.inputCoin.clicked.connect(self.savecoin) # ticckers info. update
        self.resetCoin.clicked.connect(self.resetcoin)# tickers info. reset
        self.sellCoin.clicked.connect(self.try_sell)

    def updateCoin(self):

        conn = pymysql.connect(host='db-2pm0u.pub-cdb.ntruss.com', port=3306, user='mockcoiner_dba',
                               password='sinbu753951!', db='mockcoiner_db', charset='utf8')
        curs = conn.cursor()

        sql_open = "SELECT * FROM mockcoiner_db.user WHERE email = '" + str(_id_) + "'"
        curs.execute(sql_open)

        rows = curs.fetchall()

        self.balancebtn.setText("%s" % (rows[0][4]))
        self.coin1.setText("%s" % (rows[0][5]))
        self.coin2.setText("%s" % (rows[0][6]))
        self.coin3.setText("%s" % (rows[0][7]))
        self.coin4.setText("%s" % (rows[0][8]))
        self.coin5.setText("%s" % (rows[0][9]))
        self.coin1_name.setText("%s" % (rows[0][10]))
        self.coin2_name.setText("%s" % (rows[0][11]))
        self.coin3_name.setText("%s" % (rows[0][12]))
        self.coin4_name.setText("%s" % (rows[0][13]))
        self.coin5_name.setText("%s" % (rows[0][14]))


    def MLTestingModule(self):
        dlg3 = MLDialog()
        dlg3.setStyleSheet(QSS)
        dlg3.exec_()


    def startThread(self):
        self.tworker.start()
        self.tradeBtn.setDisabled(True)
    def resetThread(self):
        self.tradeBtn.setEnabled(True)
        if self.tworker.isRunning():
            self.tworker.terminate()
            self.tworker.wait()
            self.tworker.disconnect()
            self.textEdit.append("종료되었습니다.")

    def resetcoin(self):

        conn = pymysql.connect(host='db-2pm0u.pub-cdb.ntruss.com',port=3306 ,user='mockcoiner_dba',password='sinbu753951!',db='mockcoiner_db',charset='utf8')
        curs = conn.cursor()

        sql_open = "SELECT * FROM mockcoiner_db.user WHERE email = '"+str(_id_)+"'"
        curs.execute(sql_open)

        rows = curs.fetchall()
        for i in range(5):
            if rows[0][i+5] != '0':
                self.lineEdit.setText("보유하고 있는 코인을 먼저 판매하세요.")
                continue
            else:

                del tickers[:]
                del stockList[:]
                portfolio_rate.clear()
                self.lineEdit.setText("리셋되었습니다.")

    def savecoin(self):
        if not tickers  :
            a = self.comboBox.currentText()
            tickers.append(a)
            stockList.append(a)
            a = self.comboBox2.currentText()
            tickers.append(a)
            stockList.append(a)
            a = self.comboBox3.currentText()
            tickers.append(a)
            stockList.append(a)
            a = self.comboBox4.currentText()
            tickers.append(a)
            stockList.append(a)
            a = self.comboBox5.currentText()
            tickers.append(a)
            stockList.append(a)

            global portfolio_rate

            portfolio_rate = {
                tickers[0]: [0.1],
                tickers[1]: [0.1],  
                tickers[2]: [0.1],  
                tickers[3]: [0.1],  
                tickers[4]: [0.1],  
                "cash": [0.5]  
            }
            self.lineEdit.setText(str(tickers))

            def Convert(lst):
                res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
                return res_dct
        else :
            self.lineEdit.setText("먼저 리셋해 주세요!")
    @pyqtSlot()
    def try_buy(self):
        if price is not None:
            # 시간을 저장하는 함수
            # now = 현재시각 , price = 코인 현재가격 target = 목표가격 ma = 이동평균선
            # 데이터베이스 rows 인자 별 의미 -> 0:PID 4:balance 5:unit1 6:unit2 7:unit3 8:unit4 9:unit5
            # 여기에 데이터베이스 식별을 위한 유저 개인 정보를 더 받아와야 함
            try:

                ###mysql 데이터 로드 시작

                conn = pymysql.connect(host='db-2pm0u.pub-cdb.ntruss.com',port=3306 ,user='mockcoiner_dba',password='sinbu753951!',db='mockcoiner_db',charset='utf8')
                curs = conn.cursor()

                sql_open = "SELECT * FROM mockcoiner_db.user WHERE email = '"+str(_id_)+"'"
                curs.execute(sql_open)


                rows = curs.fetchall()
                balance = rows[0][4]
                balance_temp = float(balance) * 0.15
                pid = rows[0][0]

                unit = []
                for i in range(5):
                    a = float(rows[0][i + 5])
                    unit.append(a)
                for i in range(5):
                    sql_name = "UPDATE mockcoiner_db.user SET coin" + str(i + 1) + " = '" + tickers[
                        i] + "' WHERE (pid = " + str(pid) + ");"
                    print(sql_name)
                    curs.execute(sql_name)
                    conn.commit()

                ###mysql 데이터 로드 끝
                asks = []
                sell_price = []

                #Save the Coin Name to DB


                # 현재 시장 매수,매도상황
                for i in range(5):
                    orderbook = pybithumb.get_orderbook(tickers[i])
                    asks = orderbook['asks']
                    a = asks[0]['price']
                    sell_price.append(a)
                    print(sell_price[i], "sell_price")

                    print("현재가 : ", price[i], "목표가 : ", target[i], "이동평균선", ma5[i])
                    self.textEdit.append(
                        "###" + str(rows[0][i + 10]) + "###" + "\n현재가 : " + str(price[i]) + "\n목표가 : " + str(
                            target[i]) + "\n이동평균선 : " + str(ma5[i]) + "\n")



                    # 매수 조건 활성화
                    if price[i] > target[i] and price[i] > ma5[i] and noise_df[i].bool()==True:  # 목표가격 및 이동 평균선 수치보다 높다면
                        print(i+1, "번째 코인 매수신호발생!", price[i], target[i])
                        self.textEdit.append("===%s 번째 코인 %s 매수신호발생!===\n 현재가격: %s 목표가격: %s" % (i,rows[0][i+10],price[i],target[i]) )

                        ##### 거래 코인 갯수 및 잔고 계산
                        # unit_coin = 돈 -> 코인 개수로 환산
                        unit_coin = float(balance_temp) / float(sell_price[i])
                        unit_coin = round(unit_coin, 4)

                        # 거래가 일어 났으니 그 코인 개수만큼 내 계좌에서 차감 시키는 작업
                        balance_update = float(balance) - float(sell_price[i] * unit_coin)
                        balance_update = round(balance_update, 4)  # 소수점 반올림

                        #####

                        # 이전에 이미 잔고를 코인으로 전환 하였다면, 거래가 일어나지 않아야 함, 이 모델은 하루에 두번 이상 매수,매도가 일어나지 않기 때문
                        if unit[i] > float(0):
                            print("이미 매수하였습니다.")
                            self.textEdit.append("이미 매수하였습니다.")
                            pass

                        else:
                            if balance_update < 0:  # 현재 가격과 최우선 매도가 가격과의 gap,슬리피지로 인한 잔고 부족 사태가 생기는 경우를 의미
                                print("잔액부족으로 거래가 일어나지 않았습니다.")
                                self.textEdit.append("잔액부족으로 거래가 일어나지 않았습니다")
                                pass

                            # 일반적인 경우, 즉 최초로 거래가 일어났다면 코인과 잔고를 갱신시키는 작업이 필요함.
                            else:
                                # 데이터 베이스에 코인과 잔고를 갱신 시키는 작업
                                balance = balance_update
                                print("저장될 코인 개수 : ", unit_coin, " 잔고 : ", balance)
                                self.textEdit.append("저장될 코인 개수 : %s 잔고 : %s" % ( unit_coin , balance))
                                #####
                                print("Save to DataBase...")

                                sql_unit = "UPDATE mockcoiner_db.user SET unit" + str(i + 1) + " = " + str(
                                    unit_coin) + " WHERE (pid = " + str(pid) + ");"
                                # print(sql_unit)
                                curs.execute(sql_unit)
                                conn.commit()

                                sql_balance = "UPDATE mockcoiner_db.user SET balance = " + str(
                                    balance) + " WHERE (pid = " + str(pid) + ");"
                                print(sql_balance)
                                curs.execute(sql_balance)
                                conn.commit()

                    else:
                        print(i+1, "번째 코인 매수조건아님", price[i], target[i])
                        self.textEdit.append(" %s 번째 코인 %s 매수조건 아님 \n" % (i+1, rows[0][i+10]))

            except:
                print("try buy Error")
                pass
            finally:
                ###mysql
                self.textEdit.append("갱신중입니다...\n")
                print("finally pass")
                conn.close()  ##최종적으로 연결을 닫음

    def try_sell(self):  # 여기에 팔 코인 이름과 유저 정보를 받아 오는 식으로 수정해야 한다.
        try:
            ## mysql 연결
            conn = pymysql.connect(host='db-2pm0u.pub-cdb.ntruss.com',port=3306 ,user='mockcoiner_dba',password='sinbu753951!',db='mockcoiner_db',charset='utf8')
            curs = conn.cursor()
            sql_open = "SELECT * FROM mockcoiner_db.user WHERE email = '"+str(_id_)+"'"  # 이부분을 임의로 받아와야 함
            curs.execute(sql_open)
            rows = curs.fetchall()  # 해당 유저의 모든 정보를 가져옴
            pid = rows[0][0]  # pid 기본키
            balance = rows[0][4]  # 계좌

            unit = []
            for i in range(5):
                a = float(rows[0][i + 5])
                unit.append(a)
            ## mysql 데이터 로드 끝
            # 팔아야 할 코인이 있다면
            for i in range(5):
                if unit[i] > 0:
                    price = pybithumb.get_current_price(tickers[i])  # "BTC" 부분은 try_sell에서 인자로 받아서 다른 코인으로 가능
                    price_balance = float(balance) + float(unit[i] * price)
                    # price_balance = float(balance) + float(unit * price * 2) #변화를 보기 위하여 *2 를 해주었음
                    price_balance = round(price_balance, 4)  # 소수점 반올림
                    balance = price_balance
                    # 데이터 베이스에 저장
                    sql_balance = "UPDATE mockcoiner_db.user SET balance = " + str(balance) + " WHERE (pid = " + str(
                        pid) + ");"
                    curs.execute(sql_balance)
                    conn.commit()

                    sql_unit = "UPDATE mockcoiner_db.user SET unit" + str(i + 1) + " = 0 WHERE (pid = " + str(pid) + ");"
                    curs.execute(sql_unit)
                    conn.commit()

                # 팔고 연결을 닫음

            conn.close()
            self.lineEdit.setText("판매되었습니다.")


        except:
            print("try sell error")


    def createaccount(self):
        cre_dlg = CreateDialog()
        cre_dlg.setStyleSheet(QSS)
        cre_dlg.exec_()
        # id = dlg.id
        # password = dlg.password
        # self.label.setText("id: %s password: %s" % (id, password))

    def loginevent(self):  # 버튼이 눌리면 다이얼로그가 생성됨x
        dlg = LogInDialog()
        dlg.setStyleSheet(QSS)
        dlg.exec_()
        id = dlg.id
        global _id_
        _id_ = id
        password = dlg.password
        ########sql session
        conn = pymysql.connect(host='db-2pm0u.pub-cdb.ntruss.com',port=3306 ,user='mockcoiner_dba',password='sinbu753951!',db='mockcoiner_db',charset='utf8')
        curs = conn.cursor()

        sql_open = "SELECT * FROM mockcoiner_db.user WHERE email = '"+str(id)+"'"
        curs.execute(sql_open)
        rows = curs.fetchall()
        if password == rows[0][2]:
            self.balancebtn.setText("%s" % (rows[0][4]))
            self.coin1.setText("%s" % (rows[0][5]))
            self.coin2.setText("%s" % (rows[0][6]))
            self.coin3.setText("%s" % (rows[0][7]))
            self.coin4.setText("%s" % (rows[0][8]))
            self.coin5.setText("%s" % (rows[0][9]))
            self.coin1_name.setText("%s" % (rows[0][10]))
            self.coin2_name.setText("%s" % (rows[0][11]))
            self.coin3_name.setText("%s" % (rows[0][12]))
            self.coin4_name.setText("%s" % (rows[0][13]))
            self.coin5_name.setText("%s" % (rows[0][14]))

            self.tradeBtn.setEnabled(True)
            self.closeBtn.setEnabled(True)
            self.comboBox.setEnabled(True)
            self.comboBox2.setEnabled(True)
            self.comboBox3.setEnabled(True)
            self.comboBox4.setEnabled(True)
            self.comboBox5.setEnabled(True)

            self.inputCoin.setEnabled(True)
            self.resetCoin.setEnabled(True)
            self.sellCoin.setEnabled(True)

            self.pushButton_2.setEnabled(True)
            self.pushButton_3.setEnabled(True)

            self.pushButton_5.setEnabled(True)
        else:
            self.balancebtn.setText("비밀번호 오류")

    @pyqtSlot(dict)  # this function is only finished slot that received dict type object
    def update_table_widget(self, data):
        try:
            for ticker, infos in data.items():
                index = tickers.index(ticker)

                self.tableWidget.setItem(index, 0, QTableWidgetItem(ticker))
                self.tableWidget.setItem(index, 1, QTableWidgetItem(str(infos[0])))
                self.tableWidget.setItem(index, 2, QTableWidgetItem(str(infos[1])))
                self.tableWidget.setItem(index, 3, QTableWidgetItem(str(infos[2])))
        except:
            pass

    def backTestingModule(self): #backtesting module
        dlg = BacktestingWindow()
        dlg.setStyleSheet(QSS)
        dlg.exec_()
    def backTestingMoudule_dist(self):
        dlg2 = BacktestingWindow_dist()
        dlg2.setStyleSheet(QSS)
        dlg2.exec_()

app = QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()