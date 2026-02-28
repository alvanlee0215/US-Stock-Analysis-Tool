import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_stock_analysis(ticker_symbol):
    # 1. 獲取數據
    df = yf.download(ticker_symbol, period="6mo", interval="1d")
    if df.empty:
        print("找不到標的，請檢查代碼是否正確。")
        return

    # 2. 計算波動度 (Volatility)
    df['Returns'] = df['Close'].pct_change()
    volatility = df['Returns'].std() * np.sqrt(252) * 100
    
    # 3. 費波那契位階 (Fibonacci)
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    fib_levels = {
        '0.0% (High)': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100% (Low)': low
    }

    # 4. 蒙地卡羅預測 (Monte Carlo Simulation)
    days = 30
    sims = 1000
    last_price = df['Close'].iloc[-1].item()
    
    mu = df['Returns'].mean()
    var = df['Returns'].var()
    drift = mu - (0.5 * var)
    stdev = df['Returns'].std()
    
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, sims)))
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price * daily_returns[0]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]

    # 5. 視覺化繪圖
    plt.figure(figsize=(12, 6))
    plt.plot(price_paths, color='gray', alpha=0.1)
    plt.axhline(y=last_price, color='blue', linestyle='--', label=f'Current: {last_price:.2f}')
    
    # 標示費波那契線
    colors = ['red', 'orange', 'green', 'cyan', 'purple', 'black']
    for (name, val), color in zip(fib_levels.items(), colors):
        plt.axhline(y=val, color=color, linestyle='-', alpha=0.6, label=f'Fib {name}: {val:.2f}')
    
    plt.title(f'{ticker_symbol} 30-Day Forecast & Fibonacci Levels')
    plt.xlabel('Days Forward')
    plt.ylabel('Price (USD)')
    plt.legend(loc='upper left', fontsize='small')
    plt.savefig('analysis_result.png')
    print(f"分析完成！年化波動度：{volatility:.2f}%。圖表已儲存為 analysis_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL", help="美股代碼 (例如: NVDA, TSLA)")
    args = parser.parse_args()
    get_stock_analysis(args.ticker)
