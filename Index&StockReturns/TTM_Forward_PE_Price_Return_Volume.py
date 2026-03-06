
# Enhanced Cell 2 (Corrected): Using actual queries and computing TTM/Forward PE from net profit and EPS

combined_data = []

for stock_ticker in tickers:
    # Load Annual Data (for shares o/s)
    df_pnl = pd.read_sql("SELECT ticker, date, net_profit, eps FROM stock_pnl WHERE ticker = %s ORDER BY date;", connection, params=(stock_ticker,))
    df_pnl['date'] = pd.to_datetime(df_pnl['date'])
    df_pnl['shares_os'] = df_pnl['net_profit'] / df_pnl['eps']

    # Load Quarterly Net Profit
    df_qtr = pd.read_sql("SELECT ticker, date, net_profit, result_date FROM stock_quarter WHERE ticker = %s ORDER BY date;", connection, params=(stock_ticker,))
    df_qtr['date'] = pd.to_datetime(df_qtr['date'])
    df_qtr['result_date'] = pd.to_datetime(df_qtr['result_date'])
    df_qtr['actual_date'] = np.where(df_qtr['result_date'] > pd.to_datetime('1900-01-01'), df_qtr['result_date'], df_qtr['date'])
    df_qtr['TTM_net_profit'] = df_qtr['net_profit'].rolling(window=4).sum()

    # Merge to infer latest share o/s
    latest_os = df_pnl.set_index('date')['shares_os'].resample('Q').ffill().bfill()
    df_qtr['shares_os'] = df_qtr['date'].map(lambda d: latest_os[latest_os.index <= d].iloc[-1] if not latest_os[latest_os.index <= d].empty else np.nan)

    # Compute EPS
    df_qtr['TTM_EPS'] = df_qtr['TTM_net_profit'] / df_qtr['shares_os']
    df_qtr['Fwd_EPS'] = df_qtr['TTM_EPS'].shift(-4)

    # Load Price Data
    df_price = pd.read_sql("SELECT date, close_price FROM nse_price_history WHERE nse_ticker = %s ORDER BY date;", connection, params=(stock_ticker,))
    df_price['date'] = pd.to_datetime(df_price['date'])

    # Merge with price using actual_date
    df_merge = pd.merge_asof(df_qtr.sort_values('actual_date'), df_price.sort_values('date'), left_on='actual_date', right_on='date', direction='backward')
    df_merge['ticker'] = stock_ticker
    df_merge['TTM_PE'] = df_merge['close_price'] / df_merge['TTM_EPS']
    df_merge['Fwd_PE'] = df_merge['close_price'] / df_merge['Fwd_EPS']
    combined_data.append(df_merge)

# Combine all tickers
df_all = pd.concat(combined_data)

# Plot TTM PE
fig_ttm = go.Figure()
fig_ttm.add_trace(go.Scatter(x=df_index['date'], y=df_index['pe'], mode='lines', name=f'{index_ticker} TTM PE', line=dict(color='black')))
for ticker in tickers:
    df_sub = df_all[df_all['ticker'] == ticker]
    fig_ttm.add_trace(go.Scatter(x=df_sub['actual_date'], y=df_sub['TTM_PE'], mode='lines', name=f'{ticker} TTM PE'))
fig_ttm.update_layout(
    title=f'TTM PE Chart ({", ".join(tickers)} vs {index_ticker}) - {date_range}',
    xaxis_title='Date',
    yaxis_title='PE Ratio',
    template='plotly_white',
    legend=dict(x=1.05, y=1),
    margin=dict(r=150)
)
fig_ttm.show()

# Plot Forward PE
fig_fwd = go.Figure()
fig_fwd.add_trace(go.Scatter(x=df_index['date'], y=df_index['value'] / df_index['pe'].shift(-252), mode='lines', name=f'{index_ticker} Forward PE', line=dict(color='blue')))
for ticker in tickers:
    df_sub = df_all[df_all['ticker'] == ticker]
    fig_fwd.add_trace(go.Scatter(x=df_sub['actual_date'], y=df_sub['Fwd_PE'], mode='lines', name=f'{ticker} Forward PE'))
fig_fwd.update_layout(
    title=f'Forward PE Chart ({", ".join(tickers)} vs {index_ticker}) - {date_range}',
    xaxis_title='Date',
    yaxis_title='Forward PE',
    template='plotly_white',
    legend=dict(x=1.05, y=1),
    margin=dict(r=150)
)
fig_fwd.show()
