# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:30:48 2019

@author: hwan
"""

import pybithumb
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas import Timestamp
import operator

tickers = pybithumb.get_tickers()
print(tickers)
ticker_list = ["BTC", "ETH", "BCH", "LTC","ETC"]

def realtime_noise_ror(ticker, k=0.5):
    df = pybithumb.get_ohlcv(ticker)

    # 일중 노이즈 계산
    df['noise'] = 1 - abs((df['open'] - df['close']) / (df['high'] - df['low']))
    df['noise'] = df['noise'].shift(1)
    # 20일 평균 노이즈
    df['no20'] = df['noise'].rolling(window=20).mean().shift(1)
    # 일중 변동폭 * 돌파계수 노이즈 비율로 실시간 동기화
    df['range'] = (df['high'] - df['low']) * df['no20']
    # 변동성 돌파 전략 기본 수식
    df['target'] = df['open'] + df['range'].shift(1)

    # 평균이동선
    df['ma3'] = df['close'].rolling(window=3).mean().shift(1)
    df['ma5'] = df['close'].rolling(window=5).mean().shift(1)
    df['ma7'] = df['close'].rolling(window=7).mean().shift(1)
    df['ma13'] = df['close'].rolling(window=13).mean().shift(1)

    # 평균이동선 중 하나라도 커야 거래를 시키겠다.
    df['bull5'] = ((df['open'] > df['ma3']) | (df['open'] > df['ma5']) | (df['open'] > df['ma7']) | (
                df['open'] > df['ma13']))

    # 평균 이동선에 따른 강도 조절
    df['score3'] = np.where(df['open'] > df['ma3'], 1, 0)
    df['score5'] = np.where(df['open'] > df['ma5'], 1, 0)
    df['score7'] = np.where(df['open'] > df['ma7'], 1, 0)
    df['score13'] = np.where(df['open'] > df['ma13'], 1, 0)

    # 평균이동선 개수에 따른 강도 조절
    df['score'] = (df['score3'] + df['score5'] + df['score7'] + df['score13']) / 4
    df['total'] = df['close'] * df['score'] + df['target'] * (1 - df['score'])

    # fee = 0.0032

    # ★거래 알고리즘 체크 조건이 true이고 목표가격보다 가격이 높으면 거래를 하겠다. 수익률은 목표가격 대비 구매가격으로 표시
    df['ror'] = np.where((df['high'] > df['target']) & df['bull5'],
                         df['total'] / df['target'],
                         1)

    # 누적 수익률
    df['cumprod2'] = df['ror'].cumprod()
    # MDD = 누적 수익률 대비 최대 손실율
    df['mdd2'] = (df['cumprod2'].cummax() - df['cumprod2']) / df['cumprod2'].cummax() * 100

    # plt.subplot(2,1,1)
    # plt.plot(df['close'])
    # plt.subplot(2,1,2)
    # plt.plot(df['cumprod2'],'r')
    # plt.show()

    #print("MDD: ", df['mdd2'].max())
    #print("ROR: ", df['cumprod2'][-2])

    #df.to_excel("dist_test.xlsx")

    return df


def dist_ror(ticker_list):
    # 입력 받은 리스트를 통하여 분산 투자 코인 종목 결정
    # ticker_list = ['BTC','ETH','DASH','LTC','ETC']

    # 코인 데이터 프레임 생
    df_list = []
    for i in ticker_list:
        df_list.append(realtime_noise_ror(i))

    # 분산 투자를 위한 데이터 작업
    df_list[0]['coin2'] = df_list[1]['cumprod2']
    df_list[0]['coin3'] = df_list[2]['cumprod2']
    df_list[0]['coin4'] = df_list[3]['cumprod2']
    df_list[0]['coin5'] = df_list[4]['cumprod2']

    df_list[0]['coin2_ror'] = df_list[1]['ror']
    df_list[0]['coin3_ror'] = df_list[2]['ror']
    df_list[0]['coin4_ror'] = df_list[3]['ror']
    df_list[0]['coin5_ror'] = df_list[4]['ror']

    # 결측치 제거
    df_list[0] = df_list[0].dropna()

    # 포트폴리오의 MDD 및 ROR 계산
    # print("ROR dist: ",
    #       (df_list[0]['cumprod2'][-2] +
    #        df_list[0]['coin2'][-2] +
    #        df_list[0]['coin3'][-2] +
    #        df_list[0]['coin4'][-2] +
    #        df_list[0]['coin5'][-2]) / 5)

    df_list[0]['portpolio'] = (df_list[0]['cumprod2'] + df_list[0]['coin2'] + df_list[0]['coin3'] + df_list[0][
        'coin4'] + df_list[0]['coin5']) / 5
    df_list[0]['port_MDD'] = (df_list[0]['portpolio'].cummax() - df_list[0]['portpolio']) / df_list[0][
        'portpolio'].cummax() * 100

    # 시각화
    # plt.plot(df_list[0]['cumprod2'], label=ticker_list[0])
    # plt.plot(df_list[0]['coin2'], label=ticker_list[1])
    # plt.plot(df_list[0]['coin3'], label=ticker_list[2])
    # plt.plot(df_list[0]['coin4'], label=ticker_list[3])
    # plt.plot(df_list[0]['coin5'], label=ticker_list[4])
    # plt.plot(df_list[0]['portpolio'], label='Portpolio')
    # plt.xlabel('Date')
    # plt.ylabel('ROR')
    # plt.title('Coin Backtesting')
    # plt.legend()
    # plt.show()

    #df_list[0].to_excel("dist.xlsx")
    #print("MDD: ", df_list[0]['port_MDD'].max())

    return df_list[0]


def noiseSorting(ticker_list):
    # 노이즈만 모아서 데이터 프레임을 생성
    df = noiseMake(ticker_list)
    dictionary = df.to_dict("index")

    rankedList = []
    dict_ranked_value = {}

    for timestamp in dictionary:
        coinDict = dictionary[timestamp]
        ordered_dict = sorted(coinDict.items(), key=operator.itemgetter(1))
        # operator.imtemgetter(1)는 정렬하고자 하는 키 값을 1번째 인덱스 기준으로 하겠다는 것
        dict_ranked_value[timestamp] = ordered_dict  # dictionary에 timestamp를 key로하고 딕셔너리를 value로 추가

    sortedList = sorted(dict_ranked_value.items())
    sortedDict = dict(sortedList)

    for key_ts in sortedDict:
        ListOfTuple = sortedDict[key_ts]  # timestamp를 키로하는 값 : 튜플의 리스트
        ListOfTuple = ListOfTuple[:3]  # 하위 3개만 추출하기 위해 튜플을 slice

        tempList = []
        for coin_tuple in ListOfTuple:
            tempList.append(coin_tuple[0])  # 튜플의 0번째 값. 즉, coin 종류를 추출

        sortedDict[key_ts] = tempList  # 튜플 리스트를 coin 종류 리스트로 대체

    sorted_df = pd.DataFrame(sortedDict)
    ranked_df = sorted_df.T  # transpose
    #print(ranked_df)  # output

    df['TOP1'] = ranked_df[0]
    df['TOP2'] = ranked_df[1]
    df['TOP3'] = ranked_df[2]

    # 필터링 조건 활성화
    df['coin_1'] = ((df['TOP1'] == 'coin1') | (df['TOP2'] == 'coin1') | (df['TOP3'] == 'coin1'))
    df['coin_2'] = ((df['TOP1'] == 'coin2') | (df['TOP2'] == 'coin2') | (df['TOP3'] == 'coin2'))
    df['coin_3'] = ((df['TOP1'] == 'coin3') | (df['TOP2'] == 'coin3') | (df['TOP3'] == 'coin3'))
    df['coin_4'] = ((df['TOP1'] == 'coin4') | (df['TOP2'] == 'coin4') | (df['TOP3'] == 'coin4'))
    df['coin_5'] = ((df['TOP1'] == 'coin5') | (df['TOP2'] == 'coin5') | (df['TOP3'] == 'coin5'))

    

    return df


def noiseMake(ticker_list):
    # 코인 데이터 프레임 생
    df_list = []
    for i in ticker_list:
        df_list.append(realtime_noise_ror(i))

    df = pd.DataFrame()
    df['coin1'] = df_list[0]['noise']
    df['coin2'] = df_list[1]['noise']
    df['coin3'] = df_list[2]['noise']
    df['coin4'] = df_list[3]['noise']
    df['coin5'] = df_list[4]['noise']

    df = df.dropna()

    return df


###################################
###################################
def Top3_dist_ror(ticker_list):
    # ticker_list = []
    # for i in range(5):
    #     a = input('코인을 입력하세요')
    #     ticker_list.append(a)
    #     print("입력한 것", ticker_list)

    # 분산 투자 데이터 프레임 및 노이즈 생성
    df1 = dist_ror(ticker_list)
    df2 = noiseSorting(ticker_list)

    # 데이터 프레임 병합 필터링 조건을 활성화하는 열 생성
    df1['coin_1'] = df2['coin_1']
    df1['coin_2'] = df2['coin_2']
    df1['coin_3'] = df2['coin_3']
    df1['coin_4'] = df2['coin_4']
    df1['coin_5'] = df2['coin_5']

    # 포트폴리오 top3 종목 계산
    df1['top3'] = np.where(df1['coin_1'], df1['ror'], 0)
    df1['top3'] += np.where(df1['coin_2'], df1['coin2_ror'], 0)
    df1['top3'] += np.where(df1['coin_3'], df1['coin3_ror'], 0)
    df1['top3'] += np.where(df1['coin_4'], df1['coin4_ror'], 0)
    df1['top3'] += np.where(df1['coin_5'], df1['coin5_ror'], 0)

    #필터링없이 top5로 계산한 경우 계산
    df1['top5'] = df1['ror']
    df1['top5'] += df1['coin2_ror']
    df1['top5'] += df1['coin3_ror']
    df1['top5'] += df1['coin4_ror']
    df1['top5'] += df1['coin5_ror']

    #top3와 top5 비교
    df1['top3'] = df1['top3'] / 3
    df1['top3_ror'] = df1['top3'].cumprod()

    df1['top5'] = df1['top5'] / 5
    df1['top5_ror'] = df1['top5'].cumprod()

    # plt.plot(df1['cumprod2'], label=ticker_list[0])
    # plt.plot(df1['coin2'], label=ticker_list[1])
    # plt.plot(df1['coin3'], label=ticker_list[2])
    # plt.plot(df1['coin4'], label=ticker_list[3])
    # plt.plot(df1['coin5'], label=ticker_list[4])
    # plt.plot(df1['portpolio'], label='Portpolio')
    # plt.plot(df1['top3_ror'], label='top3_ror')
    # plt.plot(df1['top5_ror'], label='top5_ror')
    # plt.xlabel('Date')
    # plt.ylabel('ROR')
    # plt.title('Coin Backtesting')
    # plt.legend()
    # plt.show()

    df1['top3_MDD'] = (df1['top3_ror'].cummax() - df1['top3_ror']) / df1['top3_ror'].cummax() * 100
    df1['top5_MDD'] = (df1['top5_ror'].cummax() - df1['top5_ror']) / df1['top5_ror'].cummax() * 100
    print("Top3 MDD: ", df1['top3_MDD'].max())
    print("Top5 MDD: ", df1['top5_MDD'].max())
    # 비교는 top3_ror과 top5_ror을 비교
    # 예상 -> top3의 ror이 더 높되 MDD는 top5가 더 높을 것이다. 이유 : 분산투자할수록 MDD는 필연적으로 낮아지기 때문. 단지, MDD를 조금만 손해보는 선에서 수익률을 극대화 할 수 있을 것이다.

    #df1.to_excel('aaaa.xlsx')
    return df1


Top3_dist_ror(ticker_list)