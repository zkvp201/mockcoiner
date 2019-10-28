# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:51:21 2019

@author: hwan
"""
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import pybithumb
from PyQt5.QtGui import *
import ror
import pybithumb
import time
import datetime
import pymysql
import numpy as np
import pandas as pd
import operator


ticker_list = ['BTC','ETH','XRP','LTC','ETC']

#time save function
def make_times(now):
    
    tomorrow = now + datetime.timedelta(1)
    #익일
    midnight = datetime.datetime(year=tomorrow.year,
                                 month=tomorrow.month,
                                 day=tomorrow.day,
                                 hour=0,
                                 minute=0,
                                 second=0)
    #10초뒤
    midnight_after_10secs = midnight + datetime.timedelta(seconds=10)
    #tiem,after 10second
    return (midnight, midnight_after_10secs)
 
#target price
def cal_target(ticker_list):
    try:
        df = ror.realtime_noise_ror(ticker_list)
        yesterday = df.iloc[-2]
 
        #coin market 24 hours
        today_open = yesterday['close']
        
        #어제 변동폭 * 돌파 계수 지정
        yesterday_high = yesterday['high']
        yesterday_low = yesterday['low']
        #돌파계수 연동을 위한 노이즈 비율
        noise = df['no20'].iloc[-1]
        
        #오늘 날의 매수 가격 설정
        target = today_open + (yesterday_high - yesterday_low) * noise
        
        noise_fillter = df['noise'][-2]        
        
        return target,noise_fillter
    except:
        return None

def noise_filtering(noise):
    name=["coin1","coin2","coin3","coin4","coin5"]    
    noise_list = dict(zip(name,noise))           
    ordered_dict = sorted(noise_list.items(), key=operator.itemgetter(1))
    ordered_dict = ordered_dict[:3]
    
    tempList=[]
    for coin_tuple in ordered_dict:
        tempList.append(coin_tuple[0])
    noise_list=tempList
    
    df=pd.DataFrame(noise_list)
    df=df.T
    df=df.rename(columns={0:'TOP1',1:'TOP2',2:'TOP3'})
    df['TOP1'] = noise_list[0]
    df['TOP2'] = noise_list[1]
    df['TOP3'] = noise_list[2]

    # 필터링 조건 활성화
    df['coin_1'] = ((df['TOP1'] == 'coin1') | (df['TOP2'] == 'coin1') | (df['TOP3'] == 'coin1'))
    df['coin_2'] = ((df['TOP1'] == 'coin2') | (df['TOP2'] == 'coin2') | (df['TOP3'] == 'coin2'))
    df['coin_3'] = ((df['TOP1'] == 'coin3') | (df['TOP2'] == 'coin3') | (df['TOP3'] == 'coin3'))
    df['coin_4'] = ((df['TOP1'] == 'coin4') | (df['TOP2'] == 'coin4') | (df['TOP3'] == 'coin4'))
    df['coin_5'] = ((df['TOP1'] == 'coin5') | (df['TOP2'] == 'coin5') | (df['TOP3'] == 'coin5'))

    df = df.T
    df = df[3:8]
    
    return df 

#이동평균선
def cal_moving_average(ticker,window=5):
    try:
        df = pybithumb.get_ohlcv(ticker)
        close = df['close']
        ma = close.rolling(window=window).mean()
        return ma[-2]
    except:
        return None
 
 
def try_buy(price, target, ma5, ticker_list, noise_df): # now = 현재시각 , price = 코인 현재가격 target = 목표가격 ma = 이동평균선
    #데이터베이스 rows 인자 별 의미 -> 0:PID 4:balance 5:unit1 6:unit2 7:unit3 8:unit4 9:unit5
    #여기에 데이터베이스 식별을 위한 유저 개인 정보를 더 받아와야 함
    try: 

        ###mysql 데이터 로드 시작
        
        conn = pymysql.connect(host='localhost',user='root',password='sinbu753951!',db='mockcoiner',charset='utf8')
        curs = conn.cursor()

        sql_open = "SELECT * FROM mockcoiner.user WHERE email = 'abcd@naver.com'"
        curs.execute(sql_open)

        rows = curs.fetchall() 
        balance = rows[0][4]
        balance_temp = float(balance) * 0.15       
        pid = rows[0][0]
        
        unit = []        
        for i in range(5):
            a = float(rows[0][i+5])
            unit.append(a)

        

        ###mysql 데이터 로드 끝
        asks = []
        sell_price = []
        #현재 시장 매수,매도상황
        for i in range(5):
            orderbook = pybithumb.get_orderbook(ticker_list[i])
            asks = orderbook['asks']
            a= asks[0]['price']
            sell_price.append(a)              
            print(sell_price[i],"sell_price")

            print("현재가 : ",price[i],"목표가 : ",target[i], "이동평균선", ma5[i] )        

            price[4]=500000
            price[3]=1000000
            target[4]=400000
            target[3]=900000
            ma5[4]=100000
            ma5[3]=100000

            #매수 조건 활성화             
            if price[i] > target[i] and price[i] > ma5[i] and noise_df.iloc[i].bool() == True : # 목표가격 및 이동 평균선 수치보다 높다면
                print( i, "번째 코인 매수신호발생!", price[i], target[i])
                
                ##### 거래 코인 갯수 및 잔고 계산
                #unit_coin = 돈 -> 코인 개수로 환산
                unit_coin = float(balance_temp)/float(sell_price[i])
                unit_coin = round(unit_coin,4)

                #거래가 일어 났으니 그 코인 개수만큼 내 계좌에서 차감 시키는 작업
                balance_update = float(balance) - float(sell_price[i] * unit_coin)
                balance_update = round(balance_update,4)#소수점 반올림                               

                
                #####

                # 이전에 이미 잔고를 코인으로 전환 하였다면, 거래가 일어나지 않아야 함, 이 모델은 하루에 두번 이상 매수,매도가 일어나지 않기 때문
                if unit[i] > float(0):
                    print("이미 매수하였습니다.")
                    pass

                else:
                    if balance_update < 0 : #현재 가격과 최우선 매도가 가격과의 gap,슬리피지로 인한 잔고 부족 사태가 생기는 경우를 의미
                        print("잔액부족으로 거래가 일어나지 않았습니다.")
                        pass

                    #일반적인 경우, 즉 최초로 거래가 일어났다면 코인과 잔고를 갱신시키는 작업이 필요함.
                    else:
                        #데이터 베이스에 코인과 잔고를 갱신 시키는 작업
                        balance = balance_update
                        print("저장될 코인 개수 : ",unit_coin," 잔고 : ",balance)
                        #####
                        print("Save to DataBase...")             

                        sql_unit = "UPDATE mockcoiner.user SET unit"+str(i+1)+" = "+ str(unit_coin) +" WHERE (pid = "+ str(pid) +");"
                        print(sql_unit)
                        curs.execute(sql_unit)
                        conn.commit()

                        sql_balance = "UPDATE mockcoiner.user SET balance = "+ str(balance) +" WHERE (pid = "+ str(pid) +");"
                        print(sql_balance)
                        curs.execute(sql_balance)
                        conn.commit()            
        
            else:
                        print( i, "번째 코인 매수조건아님", price[i], target[i])


    except:
        print("try buy Error")
        pass
    finally:
        ###mysql
        print("finally pass")

        conn.close() ##최종적으로 연결을 닫음

        ###mysql

        
 
 
def try_sell(now,ticker_list): #여기에 팔 코인 이름과 유저 정보를 받아 오는 식으로 수정해야 한다.
    try:
        #보유한 코인수 * 현재 price -> db에 업데이트하고 코인 개수를 0개로 초기화

        ## mysql 연결

        conn = pymysql.connect(host='localhost',user='root',password='sinbu753951!',db='mockcoiner',charset='utf8')
        curs = conn.cursor()

        sql_open = "SELECT * FROM mockcoiner.user WHERE email = 'abcd@naver.com'" # 이부분을 임의로 받아와야 함
        curs.execute(sql_open)

        rows = curs.fetchall() # 해당 유저의 모든 정보를 가져옴
        pid = rows[0][0] # pid 기본키
        balance = rows[0][4] #계좌        
        
        unit = []        
        for i in range(5):
            a = float(rows[0][i+5])
            unit.append(a)

        ## mysql 데이터 로드 끝

        #팔아야 할 코인이 있다면
        for i in range(5):
            if unit[i] > 0 :
                price = pybithumb.get_current_price(ticker_list[i]) #"BTC" 부분은 try_sell에서 인자로 받아서 다른 코인으로 가능
                price_balance = float(balance) + float(unit[i] * price)
                #price_balance = float(balance) + float(unit * price * 2) #변화를 보기 위하여 *2 를 해주었음
                price_balance = round(price_balance,4) # 소수점 반올림

                balance = price_balance
                #데이터 베이스에 저장
                sql_balance = "UPDATE mockcoiner.user SET balance = "+ str(balance) +" WHERE (pid = "+ str(pid) +");"
                curs.execute(sql_balance)
                conn.commit()

                sql_unit = "UPDATE mockcoiner.user SET unit"+ str(i+1) +" = 0 WHERE (pid = "+ str(pid) +");"            
                curs.execute(sql_unit)
                conn.commit()

                
            #팔고 연결을 닫음

        conn.close()

        
    except:
        print("try sell error")
 
 


################################## 함수 Body ##################################
#최초 실행시의 시간 지정
def realTrade():
    now = datetime.datetime.now()
    #익일 ,10초뒤
    time1, time2 = make_times(now)

    #다중 코인 target 설정
    #초기 실행 데이터 initialization
    target=[]
    price=[]
    ma5=[]
    noise=[]


    for i in range(5):
        target_temp,noise_temp= cal_target(ticker_list[i])
        target.append(target_temp)#목표 가격을 가져온다.
        noise.append(noise_temp)#노이즈를 가져온다.

        temp = pybithumb.get_current_price(ticker_list[i])
        price.append(temp) #현재 가격을 가져온다.

        temp = cal_moving_average(ticker_list[i],window=5)
        ma5.append(temp) # 5일 이동평균선을 가져온다.

    noise_df = noise_filtering(noise)
    print("노이즈 값",noise_df)


    #실제 거래 함수 body

        #데이터 받아오는 부분
    now = datetime.datetime.now()

    # 00:00:00 ~ 00:00:10 에는 매수와 다음 거래를 위한 데이터 기반 작업을 실시 한다.
    if time1 < now < time2:
        #try_sell(now)
        for i in range(5):
            temp = cal_moving_average(ticker_list[i],window=5)
            ma5.append(temp) #이동 평균선

            target_temp,noise_temp= cal_target(ticker_list[i])
            target.append(target_temp)
            noise.append(noise_temp)

            temp = pybithumb.get_current_price(ticker_list[i])
            price.append(temp) #현재 가격

        noise_df = noise_filtering(noise)

        time1, time2 = make_times(now) #다음날을 재지정


    #가격 확인 및 구매는 일정 주기마다 실행시키고, 함수 내에서 거래 조건을 판단한다.
    if price is not None:
        try_buy(now, price, target, ma5, ticker_list, noise_df)
        #try_sell(now, ticker_list)
    #문제점 : 매번 데이터 베이스 연결을 open 해야 됨. 거래가 일어났을 때에만 Open하는 방식으로 바꿀 수 있다면 더 좋을 것


    return target

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    