import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 建立輸出資料夾
os.makedirs('./results', exist_ok=True)

ROLLING_WINDOW = 252
MIN_PERIODS = 252
COST_RATE = 0.001425 # 交易成本
TOP_PCT = 0.3 # 做多前 30%
BOTTOM_PCT = 0.3 # 做空後 30%
GROUPING_DAYS = [5, 10, 15, 20, 'ME'] # 測試的分組日：每月第 5, 10, 15, 20 個交易日及月底

data = pd.read_csv('./Adjusted_Close.csv', index_col=0, parse_dates=True)
data.index = pd.to_datetime(data.index)
data = data.sort_index()

data = data.loc['2009-01-01':]

# 讀取台灣大盤資料作為 Benchmark
taiex = pd.read_excel('./TAIEX.xlsx', index_col=0, parse_dates=True)
taiex.index = pd.to_datetime(taiex.index)
taiex = taiex.sort_index()
taiex = taiex.loc['2009-01-01':]  # 確保大盤資料範圍與策略原始資料一致

# --- 計算大盤回撤以定義「股災」區間 (例如 TAIEX 回撤 > 15%) ---
taiex_dd_series = (taiex['收盤價(元)'] / taiex['收盤價(元)'].cummax()) - 1
crash_mask = taiex_dd_series < -0.15
crash_periods = []
if crash_mask.any():
    # 尋找連續為 True 的區間起點與終點
    diff = crash_mask.astype(int).diff().fillna(0)
    starts = crash_mask.index[diff == 1]
    ends = crash_mask.index[diff == -1]
    # 處理邊界情況
    if crash_mask.iloc[0]:
        starts = starts.insert(0, crash_mask.index[0])
    if crash_mask.iloc[-1]:
        ends = ends.append(pd.Index([crash_mask.index[-1]]))
    crash_periods = list(zip(starts, ends))

taiex_returns = taiex['收盤價(元)'].pct_change()

# 1. 計算前高 (過去一年最高價)
rolling_high = data.rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).max()
factor_df = data / rolling_high
daily_returns = data.pct_change()

# 初始化用來儲存各分組日指標的串列
summary_metrics = []
rolling_sharpe_results = {}
rolling_sortino_results = {}
rolling_beta_results = {}

# 2. 開始測試不同的分組日
for g_day in GROUPING_DAYS:
    g_day_str = str(g_day) if g_day != 'ME' else 'Month_End'
    
    print(f"\n" + "="*50)
    print(f"開始測試分組日: {g_day if g_day != 'ME' else '月底 (Month End)'}")
    print("="*50)

    # 2. 找出每個月的分組日 及進場日 (分組日 + 1)
    idx_series = pd.Series(data.index, index=data.index)
    if g_day == 'ME':
        grouping_dates = idx_series.groupby([data.index.year, data.index.month]).last()
    else:
        grouping_dates = idx_series.groupby([data.index.year, data.index.month]).apply(lambda x: x.iloc[min(g_day - 1, len(x)-1)])
        
    grouping_dates = pd.DatetimeIndex(grouping_dates.values)
    
    entry_indices = [data.index.get_loc(d) + 1 for d in grouping_dates if data.index.get_loc(d) + 1 < len(data)]
    entry_dates = data.index[entry_indices]
    grouping_dates = grouping_dates[:len(entry_dates)]

    # 將分組日與進場日印出至終端機方便查看
    #df_dates = pd.DataFrame({'Grouping_Date': grouping_dates, 'Entry_Date': entry_dates})
    #print(f"\n[{g_day_str}] 分組日與進場日清單：")
    #print(df_dates.to_string(index=False))

    # 3. 初始化存放策略每日報酬的 DataFrame
    strategy_daily_ret = pd.DataFrame(index=data.index, columns=['Winner', 'Loser', 'W_minus_L', 'Winner_Count', 'Loser_Count'], dtype=float)

    # 4. 利用向量化優化每日報酬計算
    # 提取所有分組日的因子
    mid_monthly_factor = factor_df.loc[grouping_dates]
    valid_factors_mask = mid_monthly_factor.notna().sum(axis=1) >= 10
    valid_factors = mid_monthly_factor[valid_factors_mask]
    
    # 向量化計算分位數門檻
    winner_thresh = valid_factors.quantile(1 - TOP_PCT, axis=1)
    loser_thresh = valid_factors.quantile(BOTTOM_PCT, axis=1)
    
    # 產生 boolean 遮罩
    is_winner = valid_factors.ge(winner_thresh, axis=0) & valid_factors.notna()
    is_loser = valid_factors.le(loser_thresh, axis=0) & valid_factors.notna()
    
    active_g_date = pd.Series(index=data.index, dtype='datetime64[ns]')
    cost_penalty = pd.Series(0.0, index=data.index)

    for i in range(len(grouping_dates)):
        g_date = grouping_dates[i]
        e_date = entry_dates[i]
        
        if i < len(grouping_dates) - 1:
            next_e_date = entry_dates[i+1]
        else:
            next_e_date = data.index[-1]
            if e_date >= next_e_date:
                continue

        # 持有期間的交易日 (不包含進場日當天)
        holding_mask = (data.index > e_date) & (data.index <= next_e_date)
        holding_days = data.index[holding_mask]
        
        if len(holding_days) == 0:
            continue
            
        if valid_factors_mask.iloc[i]:
            # 將每日映射到所屬的分組日
            active_g_date.loc[holding_days] = g_date
            
            # 標記成本扣除日 (期初與期末)
            first_loc = data.index.get_loc(holding_days[0])
            last_loc = data.index.get_loc(holding_days[-1])
            cost_penalty.iloc[first_loc] += COST_RATE
            cost_penalty.iloc[last_loc] += COST_RATE
            
    valid_daily_indices = active_g_date.dropna().index
    g_dates_for_daily = active_g_date.dropna().values
    
    winner_pos = pd.DataFrame(False, index=data.index, columns=data.columns)
    loser_pos = pd.DataFrame(False, index=data.index, columns=data.columns)
    
    # 一次性賦值擴張後的 Boolean Mask
    if len(valid_daily_indices) > 0:
        winner_pos.loc[valid_daily_indices] = is_winner.loc[g_dates_for_daily].values
        loser_pos.loc[valid_daily_indices] = is_loser.loc[g_dates_for_daily].values

    # 計算原始每日報酬 (利用 where 將未持倉標的過濾為 NaN，再取平均)
    w_daily_returns = daily_returns.where(winner_pos)
    l_daily_returns = daily_returns.where(loser_pos)
    
    raw_winner_ret = w_daily_returns.mean(axis=1)
    raw_loser_ret = l_daily_returns.mean(axis=1)
    
    # 扣除交易成本 (如果 raw_*_ret 為 NaN，減去 cost 仍為 NaN，可確保無交易日不受影響)
    strategy_daily_ret['Winner'] = raw_winner_ret - cost_penalty
    strategy_daily_ret['Loser'] = raw_loser_ret - cost_penalty
    strategy_daily_ret['W_minus_L'] = raw_winner_ret - raw_loser_ret - (2 * cost_penalty)
    
    strategy_daily_ret['Winner_Count'] = winner_pos.sum(axis=1).replace(0, np.nan)
    strategy_daily_ret['Loser_Count'] = loser_pos.sum(axis=1).replace(0, np.nan)

    numeric_cols = ['Winner', 'Loser', 'W_minus_L']
    valid_res = strategy_daily_ret[numeric_cols].dropna()

    # --- 績效計算 ---
    rf = 0
    ann_factor = 252 # 改為每日資料年化因子

    cum_returns = (1 + valid_res).cumprod()
    total_returns = cum_returns.iloc[-1] - 1
    cagr = (1 + total_returns) ** (ann_factor / len(valid_res)) - 1

    sharpe = (valid_res.mean() - rf) / valid_res.std() * np.sqrt(ann_factor)
    sortino = (valid_res.mean() - rf) / valid_res[valid_res < 0].std() * np.sqrt(ann_factor)

    # --- Rolling Sharpe Calculation ---
    rolling_window_size = 252 # 252-day rolling window
    rolling_mean = valid_res[numeric_cols].rolling(window=rolling_window_size, min_periods=rolling_window_size).mean()
    rolling_std = valid_res[numeric_cols].rolling(window=rolling_window_size, min_periods=rolling_window_size).std()
    rolling_std.replace(0, np.nan, inplace=True) # Avoid division by zero
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(ann_factor)
    rolling_sharpe_results[g_day_str] = rolling_sharpe.fillna(0)

    # --- Rolling Sortino Calculation ---
    downside_returns = valid_res[numeric_cols].where(valid_res[numeric_cols] < 0)
    rolling_downside_std = downside_returns.rolling(window=rolling_window_size, min_periods=1).std()
    rolling_downside_std.replace(0, np.nan, inplace=True)
    rolling_sortino = (rolling_mean / rolling_downside_std) * np.sqrt(ann_factor)
    rolling_sortino_results[g_day_str] = rolling_sortino.fillna(0)

    # --- Rolling Beta Calculation ---
    aligned_market_ret = taiex_returns.reindex(valid_res.index)
    rolling_var = aligned_market_ret.rolling(window=rolling_window_size, min_periods=rolling_window_size).var()
    rolling_var.replace(0, np.nan, inplace=True) # 避免除以 0 錯誤
    
    rolling_beta = pd.DataFrame(index=valid_res.index, columns=numeric_cols, dtype=float)
    for col in numeric_cols:
        rolling_cov = valid_res[col].rolling(window=rolling_window_size, min_periods=rolling_window_size).cov(aligned_market_ret)
        rolling_beta[col] = rolling_cov / rolling_var
        
    rolling_beta_results[g_day_str] = rolling_beta.fillna(0)

    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns / rolling_max) - 1
    mdd = drawdowns.min()

    num_periods = len(valid_res)
    avg_holding_days = (entry_dates[1:] - entry_dates[:-1]).mean().days if len(entry_dates) > 1 else 0

    avg_winner_holdings = strategy_daily_ret['Winner_Count'].dropna().mean()
    avg_loser_holdings = strategy_daily_ret['Loser_Count'].dropna().mean()

    # 轉換為月報酬以計算各年度平均月報酬率
    monthly_res = (1 + valid_res).resample('ME').apply(lambda x: x.prod() - 1 if len(x) > 0 else np.nan)
    annual_mean_returns = monthly_res.groupby(monthly_res.index.year).mean()
    annual_mean_returns = annual_mean_returns[['Winner', 'Loser', 'W_minus_L']]
    annual_mean_returns.columns = ['winner_ret_mean', 'loser_ret_mean', 'wml_ret_mean']

    print("\n====== 各年度平均月報酬率 ======")
    print((annual_mean_returns * 100).apply(lambda col: col.map(lambda x: f"{x:.2f}%")).to_string())
    print("===============================\n")

    if g_day == 'ME':
        (annual_mean_returns * 100).to_csv('./results/各年度平均月報酬率.csv', float_format='%.2f%%')

    # --- 繪圖 (包含 MDD Subplot) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # 計算大盤在相同區間的累積報酬與 MDD 以供對照
    aligned_taiex_ret = taiex_returns.reindex(valid_res.index).fillna(0)
    taiex_cum = (1 + aligned_taiex_ret).cumprod()
    taiex_dd = (taiex_cum / taiex_cum.cummax()) - 1

    top_pct_str = f"{int(TOP_PCT * 100)}%"
    bottom_pct_str = f"{int(BOTTOM_PCT * 100)}%"

    # 累計報酬圖
    cum_returns['Winner'].plot(ax=ax1, label=f'Winner (Top {top_pct_str})', color='green', linewidth=2)
    cum_returns['Loser'].plot(ax=ax1, label=f'Loser (Bottom {bottom_pct_str})', color='red', linewidth=2)
    cum_returns['W_minus_L'].plot(ax=ax1, label='L/S Hedge (W - L)', color='blue', linestyle='--', linewidth=2)
    # 將大盤設為灰色虛線，作為清晰的背景對照，避免視覺混亂
    taiex_cum.plot(ax=ax1, label='TAIEX Benchmark', color='gray', linestyle='--', alpha=0.8, linewidth=1.5)

    ax1.set_title(f'Entry Strategy ({g_day_str}) with Performance & MDD', fontsize=16)
    ax1.set_ylabel('Cumulative Wealth (Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # MDD 圖
    drawdowns['Winner'].plot(ax=ax2, label='Winner MDD', color='green', linewidth=1, alpha=0.7)
    drawdowns['Loser'].plot(ax=ax2, label='Loser MDD', color='red', linewidth=1, alpha=0.7)
    drawdowns['W_minus_L'].plot(ax=ax2, label='L/S MDD', color='blue', linestyle='--', linewidth=1, alpha=0.7)
    # taiex_dd.plot(ax=ax2, label='TAIEX MDD', color='black', alpha=0.4, linewidth=1) # 根據要求移除大盤 MDD

    ax2.fill_between(drawdowns.index, drawdowns['W_minus_L'], 0, color='blue', alpha=0.1)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 標註股災背景 (灰色區塊)
    for start, end in crash_periods:
        # 確保灰色背景只出現在策略實際運行的時間範圍內，避免 X 軸被拉得太長
        if end >= valid_res.index[0] and start <= valid_res.index[-1]:
            actual_start = max(start, valid_res.index[0])
            actual_end = min(end, valid_res.index[-1])
            ax1.axvspan(actual_start, actual_end, color='gray', alpha=0.2)
            ax2.axvspan(actual_start, actual_end, color='gray', alpha=0.2)

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


    # ... (前面的 MDD 繪圖與 Annual Average Returns 折線圖程式碼) ...

    # ==========================================
    # 繪製事件研究法 (Intramonth Trajectory) - 僅限月底進場 (ME)
    # ==========================================
    if g_day == 'ME':
        print("\n正在繪製月中累積報酬軌跡 (Event Study - Month End)...")
        
        # 1. 建立一個與 data.index 一樣長的 Series 來記錄「這是進場後的第幾天」
        relative_days = pd.Series(np.nan, index=data.index, dtype=float)
        
        for i in range(len(grouping_dates)):
            e_date = entry_dates[i]
            if i < len(grouping_dates) - 1:
                next_e_date = entry_dates[i+1]
            else:
                next_e_date = data.index[-1]
            
            # 找出這一次換股持有期間的所有交易日
            holding_mask = (data.index > e_date) & (data.index <= next_e_date)
            holding_idx = data.index[holding_mask]
            
            if len(holding_idx) > 0:
                # 賦予 1, 2, 3... 的相對天數
                relative_days.loc[holding_idx] = np.arange(1, len(holding_idx) + 1)
                
        # 2. 建立暫存 DataFrame
        # 注意：這裡刻意使用「未扣除交易成本」的 raw_ret，以便看清楚純粹的價格動能軌跡
        temp_ret = pd.DataFrame({
            'Winner': raw_winner_ret,
            'Loser': raw_loser_ret,
            'W_minus_L': raw_winner_ret - raw_loser_ret,
            'Relative_Day': relative_days
        })
        
        # 3. 去除 NaN 後，按照「進場後第幾天」分組，計算歷史平均「每日」報酬
        avg_trajectory = temp_ret.dropna().groupby('Relative_Day').mean()
        
        # 4. 計算累積軌跡 (為避免跨月天數不一的雜訊，最多取前 22 個交易日)
        max_days = min(22, int(avg_trajectory.index.max()))
        cum_trajectory = avg_trajectory.loc[:max_days].cumsum() * 100 # 轉成百分比
        
        # 5. 繪製軌跡圖
        fig_traj, ax_traj = plt.subplots(figsize=(10, 6))
        cum_trajectory['Winner'].plot(ax=ax_traj, color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
        cum_trajectory['Loser'].plot(ax=ax_traj, color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
        cum_trajectory['W_minus_L'].plot(ax=ax_traj, color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)
        
        ax_traj.set_title('Average Intramonth Return Trajectory (Month End Entry)', fontsize=16)
        ax_traj.set_xlabel('Trading Days Since Entry', fontsize=12)
        ax_traj.set_ylabel('Cumulative Return (%) - Gross of Costs', fontsize=12)
        ax_traj.axhline(0, color='black', linewidth=1)
        ax_traj.grid(True, linestyle='--', alpha=0.6)
        ax_traj.legend(loc='best')
        
        # 加上垂直輔助線，方便判斷第 5 天與第 10 天的獲利佔比
        if max_days >= 5:
            ax_traj.axvline(5, color='gray', linestyle=':', alpha=0.8)
            ax_traj.text(5.2, ax_traj.get_ylim()[0]*0.9, 'Day 5', color='gray')
        if max_days >= 10:
            ax_traj.axvline(10, color='gray', linestyle=':', alpha=0.8)
            ax_traj.text(10.2, ax_traj.get_ylim()[0]*0.9, 'Day 10', color='gray')
            
        plt.tight_layout()
        # 儲存圖表，且不執行 plt.close() 讓它在最後能被顯示出來
        fig_traj.savefig('./results/event_study_intramonth_trajectory_ME.png', dpi=300, bbox_inches='tight')

    # ==========================================

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

    # 將績效圖與 MDD 圖儲存 (僅輸出 ME)
    if g_day == 'ME':
        fig.savefig(f'./results/performance_and_mdd_{g_day_str}.png', dpi=300, bbox_inches='tight')

    # 關閉迴圈中產生的圖表，避免最後一起顯示 (Event study 圖除外)
    plt.close(fig)
    plt.close(fig3)

# ==========================================
# 五、 資料分析 (Data Analysis)
# (註：此區塊為框架保留區，待後續探索性資料分析 EDA 完成後填補)
# ==========================================
print("\n" + "="*80)
print("====== 五、 資料分析 (Data Analysis) ======")
print("="*80)

print("1. 因子分佈檢定：分析 52 週高點比率在多頭與空頭市場中的橫截面分佈（偏態與峰度變化）")
bear_mask = crash_mask.reindex(factor_df.index).fillna(False)
bull_mask = ~bear_mask

bull_factors = factor_df[bull_mask]
bear_factors = factor_df[bear_mask]

bull_skew = bull_factors.skew(axis=1).mean()
bull_kurt = bull_factors.kurtosis(axis=1).mean()
bear_skew = bear_factors.skew(axis=1).mean()
bear_kurt = bear_factors.kurtosis(axis=1).mean()

print(f"   - 多頭市場平均偏態: {bull_skew:.2f}, 平均峰度: {bull_kurt:.2f}")
print(f"   - 空頭市場平均偏態: {bear_skew:.2f}, 平均峰度: {bear_kurt:.2f}")

fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
bull_vals = bull_factors.to_numpy().flatten()
bull_vals = bull_vals[~np.isnan(bull_vals)]
bear_vals = bear_factors.to_numpy().flatten()
bear_vals = bear_vals[~np.isnan(bear_vals)]
ax_dist.hist(bull_vals, bins=50, alpha=0.5, label='Bull Market', color='green', density=True)
ax_dist.hist(bear_vals, bins=50, alpha=0.5, label='Bear Market', color='red', density=True)
ax_dist.set_title('52-Week High Factor Distribution: Bull vs Bear Markets', fontsize=14)
ax_dist.set_xlabel('Factor Value (Price / 52-Week High)', fontsize=12)
ax_dist.set_ylabel('Density', fontsize=12)
ax_dist.legend(loc='upper left')
ax_dist.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
fig_dist.savefig('./results/factor_distribution_bull_bear.png', dpi=300, bbox_inches='tight')
plt.close(fig_dist)
print("   - 已繪製因子分佈比較圖並儲存至 ./results/factor_distribution_bull_bear.png")

print("\n2. 樣本存活率 (Sample Survival Rate)：每月符合交易資格的股票檔數變化")
# 擷取最後一次分組 (Month_End) 的持倉檔數，計算每月的平均值
monthly_counts = strategy_daily_ret[['Winner_Count', 'Loser_Count']].dropna().resample('ME').mean()

# 將每月的資料依年度分組計算平均，便於終端機顯示總結
yearly_summary = monthly_counts.groupby(monthly_counts.index.year).mean()
yearly_summary.columns = ['Avg_Monthly_Winner', 'Avg_Monthly_Loser']
yearly_summary.index.name = 'Year'

print("\n[各年度平均每月符合資格檔數 (Time Series Summary)]")
print(yearly_summary.to_string(float_format='{:.1f}'.format))

# 儲存完整的月度時間序列資料至 CSV
monthly_counts.index.name = 'Month'
monthly_counts.to_csv('./results/monthly_survival_rate.csv', float_format='%.1f')
print("   - 已輸出詳細的「月度」樣本存活率表格至 ./results/monthly_survival_rate.csv")

print("="*80 + "\n")

print("\n所有測試記錄完成！")

summary_df = pd.DataFrame(summary_metrics).set_index('Grouping_Day')

summary_df.to_csv('./results/summary_df.csv', float_format='%.2f')

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
plt.savefig('./results/cagr_by_grouping_days.png', dpi=300, bbox_inches='tight')

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
plt.savefig('./results/sharpe_ratio_by_grouping_days.png', dpi=300, bbox_inches='tight')

# ==========================================
# 繪製 Rolling Sharpe Ratio 比較圖 (僅顯示月底)
# ==========================================
print("\n正在繪製 Rolling Sharpe & Sortino Ratio 比較圖 (Month End)...")

fig_sharpe, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

if 'Month_End' in rolling_sharpe_results and 'Month_End' in rolling_sortino_results:
    df_sharpe = rolling_sharpe_results['Month_End']
    df_sortino = rolling_sortino_results['Month_End']
    
    ax1.plot(df_sharpe.index, df_sharpe['Winner'], color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
    ax1.plot(df_sharpe.index, df_sharpe['Loser'], color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
    ax1.plot(df_sharpe.index, df_sharpe['W_minus_L'], color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

    ax1.set_title('252-Day Rolling Sharpe Ratio (Month End)', fontsize=16)
    ax1.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='best', fontsize=10)

    ax2.plot(df_sortino.index, df_sortino['Winner'], color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
    ax2.plot(df_sortino.index, df_sortino['Loser'], color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
    ax2.plot(df_sortino.index, df_sortino['W_minus_L'], color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

    ax2.set_title('252-Day Rolling Sortino Ratio (Month End)', fontsize=16)
    ax2.set_ylabel('Rolling Sortino Ratio', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='best', fontsize=10)

plt.tight_layout()
fig_sharpe.savefig('./results/rolling_sharpe_sortino_ratios_ME.png', dpi=300, bbox_inches='tight')

# ==========================================
# 繪製 Smoothed Rolling Sharpe Ratio 比較圖 (僅顯示月底)
# ==========================================
print("\n正在繪製 Smoothed Rolling Sharpe & Sortino Ratio 比較圖 (Month End)...")

fig_sharpe_smooth, (ax_s1, ax_s2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

if 'Month_End' in rolling_sharpe_results and 'Month_End' in rolling_sortino_results:
    smooth_window = 60  # 使用 60 天進行平滑處理 (約一季)
    df_sharpe_smooth = rolling_sharpe_results['Month_End'].rolling(window=smooth_window, min_periods=1).mean()
    df_sortino_smooth = rolling_sortino_results['Month_End'].rolling(window=smooth_window, min_periods=1).mean()
    
    ax_s1.plot(df_sharpe_smooth.index, df_sharpe_smooth['Winner'], color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
    ax_s1.plot(df_sharpe_smooth.index, df_sharpe_smooth['Loser'], color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
    ax_s1.plot(df_sharpe_smooth.index, df_sharpe_smooth['W_minus_L'], color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

    ax_s1.set_title(f'252-Day Rolling Sharpe Ratio (Month End) - {smooth_window} Days Smoothed', fontsize=16)
    ax_s1.set_ylabel('Smoothed Sharpe Ratio', fontsize=12)
    ax_s1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_s1.grid(True, linestyle='--', alpha=0.5)
    ax_s1.legend(loc='best', fontsize=10)

    ax_s2.plot(df_sortino_smooth.index, df_sortino_smooth['Winner'], color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
    ax_s2.plot(df_sortino_smooth.index, df_sortino_smooth['Loser'], color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
    ax_s2.plot(df_sortino_smooth.index, df_sortino_smooth['W_minus_L'], color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

    ax_s2.set_title(f'252-Day Rolling Sortino Ratio (Month End) - {smooth_window} Days Smoothed', fontsize=16)
    ax_s2.set_ylabel('Smoothed Sortino Ratio', fontsize=12)
    ax_s2.set_xlabel('Date', fontsize=12)
    ax_s2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_s2.grid(True, linestyle='--', alpha=0.5)
    ax_s2.legend(loc='best', fontsize=10)

plt.tight_layout()
fig_sharpe_smooth.savefig('./results/smoothed_rolling_sharpe_sortino_ratios_ME.png', dpi=300, bbox_inches='tight')

# ==========================================
# 繪製 Rolling Beta 比較圖 (僅顯示月底)
# ==========================================
print("\n正在繪製 Rolling Beta 比較圖 (Month End)...")

fig_beta, ax_beta = plt.subplots(figsize=(12, 6))
ax_beta.set_title('252-Day Rolling Beta (Month End)', fontsize=16)

if 'Month_End' in rolling_beta_results:
    df_beta = rolling_beta_results['Month_End']
    
    ax_beta.plot(df_beta.index, df_beta['Winner'], color='green', label=f'Winner (Top {top_pct_str})', linewidth=2)
    ax_beta.plot(df_beta.index, df_beta['Loser'], color='red', label=f'Loser (Bottom {bottom_pct_str})', linewidth=2)
    ax_beta.plot(df_beta.index, df_beta['W_minus_L'], color='blue', label='L/S Hedge (W - L)', linestyle='--', linewidth=2)

ax_beta.set_ylabel('Rolling Beta', fontsize=12)
ax_beta.set_xlabel('Date', fontsize=12)
ax_beta.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5) # Zero line
ax_beta.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5) # Market Beta = 1
ax_beta.grid(True, linestyle='--', alpha=0.5)
ax_beta.legend(loc='best', fontsize=10)
plt.tight_layout()
fig_beta.savefig('./results/rolling_beta_ME.png', dpi=300, bbox_inches='tight')

# 一起顯示最後的總結圖
#plt.show()