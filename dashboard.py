# 檔名: dashboard.py (版本 3.3.3 - 最終特權畢業版)

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
import time
import pandas_ta as ta
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 停用 requests 在 verify=False 時顯示的警告訊息
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# PART 1: 數據抓取與處理函式 (我們的引擎)
@st.cache_data(ttl="1d")
def load_data():
    DAYS_TO_QUERY = 90
    start_date = datetime.today()
    all_dividends_list = []
    progress_placeholder = st.empty()

    for i in range(DAYS_TO_QUERY):
        target_date_dt = start_date + timedelta(days=i)
        target_date_str = target_date_dt.strftime('%Y%m%d')
        progress_placeholder.text(f"正在查詢除權息日期: {target_date_str} ...")
        url = f"https://www.twse.com.tw/exchangeReport/TWT49U?response=json&strDate={target_date_str}&endDate={target_date_str}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
        try:
            response = requests.get(url, headers=headers, verify=False) # 植入萬能鑰匙
            if response.status_code == 200:
                json_data = response.json()
                if 'data' in json_data and json_data['data']:
                    daily_df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
                    daily_df['除權息日期'] = target_date_str
                    all_dividends_list.append(daily_df)
        except Exception as e:
            progress_placeholder.warning(f"抓取 {target_date_str} 資料時發生錯誤: {e}")
        time.sleep(0.3)
    
    progress_placeholder.empty()

    if not all_dividends_list:
        st.warning("未來三個月內查無任何除權息資料。")
        return pd.DataFrame()
    
    dividends_df = pd.concat(all_dividends_list, ignore_index=True)
    dividends_df = dividends_df[dividends_df['股票代號'].str.match(r'^\d{4}$|^\d{6}$')].copy()
    stock_list = dividends_df['股票代號'].unique()
    stock_data_list = []
    
    progress_bar = st.progress(0, text="正在抓取股價與計算技術指標...")
    for i, stock_id in enumerate(stock_list):
        stock_id_yf = f"{stock_id}.TW"
        try:
            stock = yf.Ticker(stock_id_yf)
            hist = stock.history(period="6mo", auto_adjust=False)
            if hist.empty or len(hist) < 26: continue
            hist.ta.macd(append=True); hist.ta.stoch(append=True)
            latest_data = hist.iloc[-1]
            stock_data_list.append({
                '股票代號': stock_id, '最新收盤價': latest_data['Close'],
                '5日均量': hist['Volume'].iloc[-5:].mean(),
                'MACD': latest_data.get('MACD_12_26_9', np.nan),
                'K值': latest_data.get('STOCHk_9_3_3', np.nan),
                'D值': latest_data.get('STOCHd_9_3_3', np.nan)
            })
        except Exception:
            pass
        time.sleep(0.2)
        progress_bar.progress((i + 1) / len(stock_list), text=f"正在處理: {stock_id_yf}")
    
    progress_bar.empty()
    price_volume_df = pd.DataFrame(stock_data_list)
    final_df = pd.merge(dividends_df, price_volume_df, on='股票代號', how='left')
    final_df.dropna(subset=['最新收盤價'], inplace=True)
    final_df.sort_values(by='除權息日期', inplace=True)
    final_df.drop_duplicates(subset=['股票代號'], keep='first', inplace=True)
    return final_df

@st.cache_data(ttl="1d")
def get_stock_history(stock_id):
    stock_id_yf = f"{stock_id}.TW"
    stock = yf.Ticker(stock_id_yf)
    hist = stock.history(period="1y", auto_adjust=False)
    if not hist.empty:
        hist.ta.macd(append=True)
        hist.ta.stoch(append=True)
        hist.ta.sma(length=5, append=True)
        hist.ta.sma(length=20, append=True)
    return hist

def plot_stock_chart(hist_df):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.15, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'], low=hist_df['Low'], close=hist_df['Close'], name="K線"), row=1, col=1)
    if 'SMA_5' in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_5'], name="5日均線", line=dict(color='orange', width=1)), row=1, col=1)
    if 'SMA_20' in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_20'], name="20日均線", line=dict(color='blue', width=1)), row=1, col=1)
    if 'Volume' in hist_df.columns:
        fig.add_trace(go.Bar(x=hist_df.index, y=hist_df['Volume'], name="成交量"), row=2, col=1)
    if 'MACD_12_26_9' in hist_df.columns and 'MACDs_12_26_9' in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MACD_12_26_9'], name="MACD", line=dict(color='purple')), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MACDs_12_26_9'], name="Signal", line=dict(color='cyan')), row=3, col=1)
    if 'STOCHk_9_3_3' in hist_df.columns and 'STOCHd_9_3_3' in hist_df.columns:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['STOCHk_9_3_3'], name="K值", line=dict(color='green')), row=4, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['STOCHd_9_3_3'], name="D值", line=dict(color='red')), row=4, col=1)
    fig.update_layout(height=800, title_text="個股歷史線圖與技術指標", xaxis_rangeslider_visible=False)
    return fig

# PART 2: 儀表板介面佈局 (最終畢業版)
st.set_page_config(layout="wide")
st.title("📊 全自動台股除權息戰情室")
st.write(f"數據最後更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

final_df = load_data()

st.sidebar.header("⚙️ 個人化篩選條件")
if st.sidebar.button("強制刷新數據"):
    st.cache_data.clear(); st.rerun()

price_lower = st.sidebar.number_input("股價下限 (元)", min_value=0, value=10)
price_upper = st.sidebar.number_input("股價上限 (元)", min_value=0, value=200)
volume_threshold = st.sidebar.number_input("5日均量下限 (張)", min_value=0, value=100)

if not final_df.empty:
    condition_price = final_df['最新收盤價'].between(price_lower, price_upper)
    condition_volume = final_df['5日均量'] > (volume_threshold * 1000)
    condition_macd = final_df['MACD'] > 0
    filtered_df = final_df[condition_price & condition_volume & condition_macd]

    st.header("🌟 黃金篩選名單")
    if filtered_df.empty:
        st.warning("提醒：目前在已公告的除權息清單中，沒有任何股票符合您的所有篩選條件。")
    else:
        sorted_df = filtered_df.sort_values(by=['除權息日期', '5日均量'], ascending=[True, False]).reset_index(drop=True)
        st.dataframe(sorted_df)

        st.header("📈 個股線圖查詢")
        stock_options = sorted_df['股票代號'].tolist()
        if stock_options:
            selected_stock = st.selectbox("請從黃金名單中選擇一檔股票：", options=stock_options)
            if selected_stock:
                hist_df = get_stock_history(selected_stock)
                if not hist_df.empty:
                    st.plotly_chart(plot_stock_chart(hist_df), use_container_width=True)
                else:
                    st.error(f"無法獲取 {selected_stock} 的歷史資料。")

    with st.expander("顯示未來90天內「已公告」的完整除權息總表"):
        st.dataframe(final_df)
else:
    st.info("目前證交所尚未公告任何未來除權息資料，或數據加載失敗。")
