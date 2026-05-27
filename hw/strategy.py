import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置tushare（免费版直接用）
ts.set_token('')
pro = ts.pro_api()

# 1. 获取贵州茅台数据（2022-01-01 到 2023-01-01）
df = pro.daily(ts_code='600519.SH', start_date='20220101', end_date='20230101')
df = df.sort_values('trade_date')
df['close'] = df['close'].astype(float)

# 2. 计算双均线
df['ma5'] = df['close'].rolling(5).mean()
df['ma20'] = df['close'].rolling(20).mean()

# 3. 生成交易信号
df['signal'] = 0
df.loc[df['ma5'] > df['ma20'], 'signal'] = 1   # 金叉买入
df.loc[df['ma5'] < df['ma20'], 'signal'] = -1  # 死叉卖出

# 4. 计算策略收益
df['return'] = df['close'].pct_change()
df['strategy_return'] = df['return'] * df['signal'].shift(1)
df['cum_return'] = (1 + df['return']).cumprod()
df['strategy_cum_return'] = (1 + df['strategy_return']).cumprod()

# 5. 统计回测结果
total_return = df['strategy_cum_return'].iloc[-1] - 1
max_drawdown = (df['strategy_cum_return'] / df['strategy_cum_return'].cummax() - 1).min()
print(f"策略总收益率: {total_return:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")

# 6. 画图对比
plt.figure(figsize=(12, 6))
plt.plot(df['trade_date'], df['cum_return'], label='标的收益', color='blue')
plt.plot(df['trade_date'], df['strategy_cum_return'], label='双均线策略收益', color='red')
plt.legend()
plt.title('双均线策略回测结果（2022-2023）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('strategy_result.png')
plt.show()