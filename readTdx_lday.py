# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 12:16:56 2017

@author: Administrator

readTdx_lday.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib  inline

import talib
import math



def read_tdxlday_2csv(dirname, fname, targetDir):
    import os
    from struct import unpack, pack
    outfile=open(dirname + os.sep + fname, 'rb')
    buf = outfile.read()
    outfile.close()
    
    inputfile=open(targetDir + os.sep + fname + '.csv', 'w')
    num=len(buf)
    numrec=num/32
    b=0
    e=32
    line=''
    linename=u'date, open, high, low, close, amount, volume, str07' + '\n'
    
    inputfile.write(linename)
    for i in xrange(numrec):
        a=unpack('IIIIIfII', buf[b:e])
        #   结尾的"\" 表示续行
        line=str(a[0])+', ' +str(a[1]/100.0)+', ' +str(a[2]/100.0)+', ' +str(a[3]/100.00)+', '     +  \
             str(a[4]/100.0)+', ' +str(a[5]/1.0)+', ' +str(a[6]/100.0)+', ' +str(a[7])+'' +'\n'
        inputfile.write(line)
        b+=32
        e+=32
    inputfile.close()
        
        

#%
def read_lday_csvFile(fname):
    import pandas as pd
    u'''read joinquant export csv file to dataframe
    param::
        fname
    return::
		4个对象的元组, (ohlc数据框, 代码, roc1序列,收益率序列)
        df, ticker, roc1, BnH
    example::
        fname= "d:/db/downloads/150195.csv"
        ohlc,ticker,roc1,BnH = read_lday_csvFile(fname)
    '''
#    fname = "d:/db/downloads/159902.csv"
    ohlc = pd.read_csv(fname, index_col=0, parse_dates=True)
#    ohlc_copy = ohlc.loc[:,:]
#    num_samples = len(ohlc)
##    ohlc.iloc[:,(0,1,2,3)].plot()
#    #df= randprice(240,20)
#    
#    n_samples = num_samples - 100  #1500
#    ohlc = ohlc_copy[-n_samples:]
    ohlc.columns = ohlc.columns.str.replace(u' open', 'O')
    ohlc.columns = ohlc.columns.str.replace(u' high', 'H')
    ohlc.columns = ohlc.columns.str.replace(u' low', 'L')
    ohlc.columns = ohlc.columns.str.replace(u' close', 'C')
    ohlc.columns = ohlc.columns.str.replace(u' amount', 'A')
    ohlc.columns = ohlc.columns.str.replace(u' volume', 'V')
    
    ohlc.index.name='tdx_DateTime'
    
    roc1 = ohlc.C.pct_change().fillna(value=0);  roc1 = roc1.rename('ROC1')
    bnh = (1.0+roc1).cumprod()
    bnh = bnh.rename('BnH')
    ticker = fname.split('/')[-1][0:8]
    ticker = ticker.upper()
    
    return ohlc, ticker, roc1, bnh
    


def HSL(vol, n):
    u'''计算移动换手率    
    '''
    out = vol/talib.SUM(vol,n)
    # 处理nan: 前面的n-1个nan values
    for i in np.arange(n-1):
        out[i]=vol[i]/vol.cumsum()[i]
    return out
            

def DMA(p,f):
    u'''
    功能:
      计算加权移动均值
    输入:    
      p: ndarray类型, 输入数据
      f: ndarray类型, 加权因子
    返回值: 
      ndarray类型
    '''
    y=[0]*len(p)
    for i in np.arange(len(p)):
        if i==0:
            y[i] = p[i]
        else:
            y[i] = f[i]*p[i] + (1-f[i])*y[i-1]
    return np.array(y)


#%%
"""This cell defineds the plot_candles function"""

def plot_candles(pricing, title=None, volume_bars=False, color_function=None, technicals=None, savefig=False):
    """ Plots a candlestick chart using quantopian pricing data.
    
    Author: Daniel Treiman
    
    Args:
      pricing: A pandas dataframe with columns ['open_price', 'close_price', 'high', 'low', 'volume']
      title: An optional title for the chart
      volume_bars: If True, plots volume bars
      color_function: A function which, given a row index and price series, returns a candle color.
      technicals: A list of additional data series to add to the chart.  Must be the same length as pricing.
    """
    def default_color(index, open_price, close_price, low, high):
        return 'black' if open_price[index] > close_price[index] else 'white' # r g b  cyan black white
    
    color_function = color_function or default_color
    technicals = technicals or []
    open_price = pricing['O']
    close_price = pricing['C']
    low = pricing['L']
    high = pricing['H']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    
    if volume_bars:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]} )
    else:
        fig, ax1 = plt.subplots(1, 1)
    fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    fig.set_size_inches(len(pricing)/20.0, 6)
    
    if title:
        ax1.set_title(title)
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
#    candles = ax1.bar(x, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
#    lines = ax1.vlines(x + 0.4, low, high, color=candle_colors, linewidth=1)
    candles = ax1.bar(x-0.4, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0.2, edgecolor='black')
    lines = ax1.vlines(x,    oc_max, high, color=['black']*len(pricing), linewidth=0.2)
    lines = ax1.vlines(x,    low, oc_min,  color=['black']*len(pricing), linewidth=0.2)
    ax1.xaxis.grid(False)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    # Assume minute frequency if first two bars are in the same day.
    frequency = 'minute' if (pricing.index[1] - pricing.index[0]).days == 0 else 'day'
    time_format = '%Y-%m-%d'
    if frequency == 'minute':
        time_format = '%H:%M'
    # Set X axis tick labels.
    x5 = np.arange(0,len(pricing), 5)
    dt_index5 = [pricing.index[i]  for i in x5]
    plt.xticks(x5, [date.strftime(time_format) for date in dt_index5], rotation=45)
    #plt.xticks(x, [date.strftime(time_format) for date in pricing.index], rotation='vertical')
    
    for indicator in technicals:
        ax1.plot(x, indicator, 'o-', linewidth=0.1, markersize=0.7, markeredgewidth=0.1) #带圆圈标记的实线
#     if dma61 in technicals: #.name = 'dma61': ax1.plot(x, indicator, linewidth=2)
#         ax1.plot(x, dma61, 'o-', color='green', linewidth=0.1, markersize=0.7, markeredgecolor='green', markeredgewidth=0.1) 
    
    if volume_bars:
        volume = pricing['V']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1000000:
            volume_scale = 'M'
            scaled_volume = volume / 1000000
        elif volume.max() > 1000:
            volume_scale = 'K'
            scaled_volume = volume / 1000
        ax2.bar(x-0.4, scaled_volume, color=candle_colors, linewidth=0.2, edgecolor='black')
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        ax2.set_title(volume_title)
        ax2.xaxis.grid(False)
        
    if savefig:
        fig.savefig('test2png.png', dpi=300)
        


#%%
# 读取csv
fname='d:/temp/python_data_gupiao/lday/sz000911.day.csv'
ohlc,ticker,roc1,bnh = read_lday_csvFile(fname) #ohlc = pd.read_csv(fname, index_col=0, parse_dates=True)
bnh = (1.0+roc1).cumprod();    bnh = bnh.rename('BnH')


#%%
# 截取一个子集, 然后计算指标, 然后画图
last60 = ohlc[-250:]

sma20 = talib.SMA(last60.C.values, 20)
p = (3*last60.H.values + 2*last60.C.values + last60.O.values + last60.L.values)/7.0
hsl61 = HSL(last60.V.values, 61)
dma61 = DMA(p, hsl61)
fc8   = talib.LINEARREG(last60.C.values, 8)

plot_candles(last60, volume_bars=True, technicals=[sma20, dma61], savefig=True)
#plot_candles(last60, volume_bars=True, technicals=[sma20, dma61, fc8], savefig=False)


