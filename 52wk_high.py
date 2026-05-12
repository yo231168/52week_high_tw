import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROLLING_WINDOW = 252
MIN_PERIODS = 252
COST_RATE = 0.001425 # 交易成本
TOP_PCT = 0.3        # 做多前 30%
BOTTOM_PCT = 0.3     # 做空後 30%
GROUPING_DAYS = [5, 10, 15, 20, 25, 'ME'] # 測試的分組日：每月第 5, 10, 15, 20, 25 個交易日及月底

data = pd.read_csv('./Adjusted_Close.csv', index_col=0, parse_dates=True)
data.index = pd.to_datetime(data.index)
data = data.sort_index()

data = data.loc['2009-01-01':]

# 1. 計算前高 (過去一年最高價)
rolling_high = data.rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).max()
factor_df = data / rolling_high

# Top / Bottom 投組報酬計算函數
def evaluate_strategy(group):
    group = group.dropna()
    if len(group) < 10: 
        return pd.Series({'Winner': np.nan, 'Loser': np.nan, 'W_minus_L': np.nan, 'Winner_Count': np.nan, 'Loser_Count': np.nan})
    
    winner_threshold = group['Factor'].quantile(1 - TOP_PCT)
    loser_threshold = group['Factor'].quantile(BOTTOM_PCT)
    
    winners = group[group['Factor'] >= winner_threshold]
    losers = group[group['Factor'] <= loser_threshold]
    
    winner_gross = winners['Next_Ret'].mean()
    loser_gross = losers['Next_Ret'].mean()
    
    # 每個月換倉的雙邊交易成本 (1 個月組合)
    monthly_cost = 2 * COST_RATE
    
    winner_net = winner_gross - monthly_cost
    loser_net = loser_gross - monthly_cost
    w_minus_l_net = winner_gross - loser_gross - (2 * monthly_cost)
    
    return pd.Series({
        'Winner': winner_net,
        'Loser': loser_net,
        'W_minus_L': w_minus_l_net,
        'Winner_Count': len(winners),
        'Loser_Count': len(losers)
    })

# 初始化用來儲存各分組日指標的串列
summary_metrics = []
rolling_sharpe_results = {}

# 2. 開始測試不同的分組日
for g_day in GROUPING_DAYS:
    g_day_str = str(g_day) if g_day != 'ME' else 'Month_End'
    
    print(f"\n" + "="*50)
    print(f"開始測試分組日: {g_day if g_day != 'ME' else '月底 (Month End)'}")
    print("="*50)

    # 2. 找出每個月的分組日 及進場日 (分組日 + 1)
    if g_day == 'ME':
        grouping_dates = data.groupby([data.index.year, data.index.month]).apply(lambda x: x.index[-1])
    else:
        grouping_dates = data.groupby([data.index.year, data.index.month]).apply(lambda x: x.index[min(g_day - 1, len(x)-1)])
        
    grouping_dates = pd.DatetimeIndex(grouping_dates.values)
    
    entry_indices = [data.index.get_loc(d) + 1 for d in grouping_dates if data.index.get_loc(d) + 1 < len(data)]
    entry_dates = data.index[entry_indices]
    grouping_dates = grouping_dates[:len(entry_dates)]

    # 3. 取得分組日的 Factor 及進場日的價格
    mid_monthly_factor = factor_df.loc[grouping_dates]
    entry_prices = data.loc[entry_dates]

    # 4. 計算下期報酬 (改為不預扣成本的毛報酬 Gross Return)
    next_period_return = (entry_prices.shift(-1) / entry_prices) - 1
    next_period_return.index = grouping_dates 

    # 5. 轉換為長表格以便分組
    factor_long = mid_monthly_factor.stack().reset_index()
    factor_long.columns = ['Date', 'Ticker', 'Factor']
    
    return_long = next_period_return.stack().reset_index()
    return_long.columns = ['Date', 'Ticker', 'Next_Ret']
    
    backtest_df = pd.merge(factor_long, return_long, on=['Date', 'Ticker'])

    # 6. 計算策略報酬
    strategy_results = backtest_df.groupby('Date').apply(evaluate_strategy)

    numeric_cols = ['Winner', 'Loser', 'W_minus_L']
    valid_res = strategy_results[numeric_cols].dropna()

    # --- 績效計算 ---
    rf = 0
    ann_factor = 12 # 月度資料

    cum_returns = (1 + valid_res).cumprod()
    total_returns = cum_returns.iloc[-1] - 1
    cagr = (1 + total_returns) ** (ann_factor / len(valid_res)) - 1

    sharpe = (valid_res.mean() - rf) / valid_res.std() * np.sqrt(ann_factor)
    sortino = (valid_res.mean() - rf) / valid_res[valid_res < 0].std() * np.sqrt(ann_factor)

    # --- Rolling Sharpe Calculation ---
    rolling_window_size = 12 # 12-month rolling window
    rolling_mean = valid_res[numeric_cols].rolling(window=rolling_window_size, min_periods=rolling_window_size).mean()
    rolling_std = valid_res[numeric_cols].rolling(window=rolling_window_size, min_periods=rolling_window_size).std()
    rolling_std.replace(0, np.nan, inplace=True) # Avoid division by zero
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(ann_factor)
    rolling_sharpe_results[g_day_str] = rolling_sharpe.fillna(0)

    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns / rolling_max) - 1
    mdd = drawdowns.min()

    num_periods = len(valid_res)
    avg_holding_days = (entry_dates[1:] - entry_dates[:-1]).mean().days if len(entry_dates) > 1 else 0

    avg_winner_holdings = strategy_results['Winner_Count'].mean()
    avg_loser_holdings = strategy_results['Loser_Count'].mean()

    annual_mean_returns = valid_res.groupby(valid_res.index.year).mean()
    annual_mean_returns = annual_mean_returns[['Winner', 'Loser', 'W_minus_L']]
    annual_mean_returns.columns = ['winner_ret_mean', 'loser_ret_mean', 'wml_ret_mean']

    print("\n====== 各年度平均月報酬率 ======")
    print((annual_mean_returns * 100).apply(lambda col: col.map(lambda x: f"{x:.2f}%")).to_string())
    print("===============================\n")

    # --- 繪圖 (包含 MDD Subplot) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    top_pct_str = f"{int(TOP_PCT * 100)}%"
    bottom_pct_str = f"{int(BOTTOM_PCT * 100)}%"

    # 累計報酬圖
    cum_returns['Winner'].plot(ax=ax1, label=f'Winner (Top {top_pct_str})', color='green', linewidth=2)
    cum_returns['Loser'].plot(ax=ax1, label=f'Loser (Bottom {bottom_pct_str})', color='red', linewidth=2)
    cum_returns['W_minus_L'].plot(ax=ax1, label='L/S Hedge (W - L)', color='blue', linestyle='--', linewidth=2)

    ax1.set_title(f'Entry Strategy ({g_day_str}) with Performance & MDD', fontsize=16)
    ax1.set_ylabel('Cumulative Wealth (Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # MDD 圖
    drawdowns['Winner'].plot(ax=ax2, label='Winner MDD', color='green', linewidth=1, alpha=0.7)
    drawdowns['Loser'].plot(ax=ax2, label='Loser MDD', color='red', linewidth=1, alpha=0.7)
    drawdowns['W_minus_L'].plot(ax=ax2, label='L/S MDD', color='blue', linestyle='--', linewidth=1, alpha=0.7)

    ax2.fill_between(drawdowns.index, drawdowns['W_minus_L'], 0, color='blue', alpha=0.1)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # --- 繪圖 (各年度平均月報酬率折線圖) ---
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    
    # 將數值轉為百分比繪圖
    plot_data = annual_mean_returns * 100
    plot_data.plot(kind='line', ax=ax3, marker='o', color=['green', 'red', 'blue'], linewidth=2, alpha=0.8)

    ax3.set_title(f'Annual Average Monthly Returns Trend by Group ({g_day_str})', fontsize=16)
    ax3.set_ylabel('Average Monthly Return (%)', fontsize=12)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.legend([f'Winner (Top {top_pct_str})', f'Loser (Bottom {bottom_pct_str})', 'L/S Hedge (W - L)'], loc='best')
    ax3.axhline(0, color='black', linewidth=1.2)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xticks(plot_data.index)
    plt.tight_layout()

    # 儲存每次迴圈的指標結果以便最後繪圖
    summary_metrics.append({
        'Grouping_Day': g_day_str,
        'Total_Ret_Winner': total_returns['Winner'],
        'Total_Ret_Loser': total_returns['Loser'],
        'Total_Ret_W_minus_L': total_returns['W_minus_L'],
        'CAGR_Winner': cagr['Winner'],
        'CAGR_Loser': cagr['Loser'],
        'CAGR_W_minus_L': cagr['W_minus_L'],
        'MDD_Winner': mdd['Winner'],
        'MDD_Loser': mdd['Loser'],
        'MDD_W_minus_L': mdd['W_minus_L'],
        'Sharpe_Winner': sharpe['Winner'],
        'Sharpe_Loser': sharpe['Loser'],
        'Sharpe_W_minus_L': sharpe['W_minus_L'],
        'Sortino_Winner': sortino['Winner'],
        'Sortino_Loser': sortino['Loser'],
        'Sortino_W_minus_L': sortino['W_minus_L'],
        'Num_Periods': num_periods,
        'Avg_Holding_Days': avg_holding_days,
        'Avg_Winner_Holdings': avg_winner_holdings,
        'Avg_Loser_Holdings': avg_loser_holdings
    })
    
    # 如果是月底 (ME)，則將圖表輸出儲存並保留在畫面上，其餘天數則關閉避免洗版
    if g_day == 'ME':
        fig.savefig('ME_cum_returns_and_MDD.png', dpi=300, bbox_inches='tight')
    else:
        plt.close(fig)
        
    plt.close(fig3)

print("\n所有測試記錄完成！")

summary_df = pd.DataFrame(summary_metrics).set_index('Grouping_Day')

print("\n" + "="*80)
print("====== 策略績效總結報告 ======")
print("="*80)

# --- Total Returns ---
df = summary_df[[c for c in summary_df.columns if 'Total_Ret' in c]]
df.columns = ['Winner', 'Loser', 'W_minus_L']
print("\n[總報酬率 (Total Returns)]")
print(df.to_string(float_format=lambda x: f"{x:.2%}"))

# --- CAGR ---
df = summary_df[[c for c in summary_df.columns if 'CAGR' in c]]
df.columns = ['Winner', 'Loser', 'W_minus_L']
print("\n[年化報酬率 (CAGR)]")
print(df.to_string(float_format=lambda x: f"{x:.2%}"))

# --- MDD ---
df = summary_df[[c for c in summary_df.columns if 'MDD' in c]]
df.columns = ['Winner', 'Loser', 'W_minus_L']
print("\n[最大回撤 (Max Drawdown)]")
print(df.to_string(float_format=lambda x: f"{x:.2%}"))

# --- Sharpe ---
df = summary_df[[c for c in summary_df.columns if 'Sharpe' in c]]
df.columns = ['Winner', 'Loser', 'W_minus_L']
print("\n[夏普比率 (Sharpe Ratio)]")
print(df.to_string(float_format='{:.2f}'.format))

# --- Sortino ---
df = summary_df[[c for c in summary_df.columns if 'Sortino' in c]]
df.columns = ['Winner', 'Loser', 'W_minus_L']
print("\n[索提諾比率 (Sortino Ratio)]")
print(df.to_string(float_format='{:.2f}'.format))

# --- Other Stats ---
stats_df = summary_df[['Num_Periods', 'Avg_Holding_Days', 'Avg_Winner_Holdings', 'Avg_Loser_Holdings']]
print("\n[其他統計]")
print(stats_df.to_string(formatters={
    'Avg_Holding_Days': '{:.0f}'.format,
    'Avg_Winner_Holdings': '{:.1f}'.format,
    'Avg_Loser_Holdings': '{:.1f}'.format
}))

print("\n" + "="*80)

# ==========================================
# 繪製所有分組日的 CAGR 比較圖
# ==========================================
top_pct_str = f"{int(TOP_PCT * 100)}%"
bottom_pct_str = f"{int(BOTTOM_PCT * 100)}%"

plt.figure(figsize=(10, 6))
# 將數值轉換成百分比進行繪圖
plt.plot(summary_df.index, summary_df['CAGR_Winner'] * 100, marker='o', color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
plt.plot(summary_df.index, summary_df['CAGR_Loser'] * 100, marker='o', color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
plt.plot(summary_df.index, summary_df['CAGR_W_minus_L'] * 100, marker='o', color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

plt.title('CAGR by Grouping Days', fontsize=16)
plt.xlabel('Grouping Days', fontsize=12)
plt.ylabel('CAGR (%)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.axhline(0, color='black', linewidth=1.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# ==========================================
# 繪製所有分組日的 Sharpe Ratio 比較圖
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(summary_df.index, summary_df['Sharpe_Winner'], marker='o', color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
plt.plot(summary_df.index, summary_df['Sharpe_Loser'], marker='o', color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
plt.plot(summary_df.index, summary_df['Sharpe_W_minus_L'], marker='o', color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

plt.title('Sharpe Ratio by Grouping Days', fontsize=16)
plt.xlabel('Grouping Days', fontsize=12)
plt.ylabel('Sharpe Ratio', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.axhline(0, color='black', linewidth=1.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# ==========================================
# 繪製 Rolling Sharpe Ratio 比較圖 (僅顯示月底)
# ==========================================
print("\n正在繪製 Rolling Sharpe Ratio 比較圖 (Month End)...")

fig_sharpe, ax = plt.subplots(figsize=(12, 6))
ax.set_title('12-Month Rolling Sharpe Ratio (Month End)', fontsize=16)

if 'Month_End' in rolling_sharpe_results:
    df_sharpe = rolling_sharpe_results['Month_End']
    
    ax.plot(df_sharpe.index, df_sharpe['Winner'], color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
    ax.plot(df_sharpe.index, df_sharpe['Loser'], color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
    ax.plot(df_sharpe.index, df_sharpe['W_minus_L'], color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', fontsize=10)
plt.tight_layout()

# 一起顯示最後的總結圖
plt.show()