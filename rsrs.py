import akshare as ak
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(layout="wide")

# 指数代码映射表
index_map = {
    "中证1000": "sh000852",
    "沪深300": "sh000300",
    "上证50": "sh000016",
    "上证指数": "sh000001",
    "北证50": "bj899050",
    "科创综指": "sh000680",
    "恒生指数": "HSI",
    "恒生科技指数": "HSTECH"
}


def fetch_index_data(index_code):
    symbol = index_map.get(index_code)
    if not symbol:
        return pd.DataFrame()

    try:
        # 区分A股和港股接口
        if symbol in ["HSI", "HSTECH"]:
            df = ak.stock_hk_daily(symbol=symbol)
        else:
            df = ak.stock_zh_index_daily(symbol=symbol)

        # 统一列名和日期格式
        df = df.rename(columns={'date': 'trade_date', 'close': 'close'})
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 计算20日均线
        df['ma20'] = df['close'].rolling(window=20).mean()

        # 检查最新数据日期
        last_date = df['trade_date'].max().strftime("%Y-%m-%d")
        current_date = datetime.now().strftime("%Y-%m-%d")
        if last_date < current_date:
            st.warning(f"最新数据截止至 {last_date}，正在尝试补充当日数据...")
        return df[['trade_date', 'open', 'high', 'low', 'close', 'ma20']]

    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        return pd.DataFrame()


def calculate_beta_and_signals(data, N, M, buy_thre, sell_thre, use_ma_filter):
    data2 = pd.DataFrame()

    if len(data) < M:
        st.error(f"数据不足！需要 {M} 条，当前 {len(data)} 条")
        return data2, buy_thre, sell_thre

    try:
        # 数据预处理
        data = data.ffill().bfill().dropna().reset_index(drop=True)
        data['pct'] = data['close'] / data['close'].shift(1) - 1.0
        data = data.dropna().reset_index(drop=True)

        # 性能优化：使用向量化操作计算滚动Beta值
        data_values = data[['open', 'high', 'low', 'close']].values
        betas = np.full(len(data), np.nan)  # 预分配数组

        # 使用向量化窗口计算
        for i in range(N - 1, len(data)):
            window = data_values[i - N + 1:i + 1]
            low_vals = window[:, 2]
            high_vals = window[:, 1]
            cov_matrix = np.cov(low_vals, high_vals)
            if cov_matrix[0, 0] != 0:
                betas[i] = cov_matrix[0, 1] / cov_matrix[0, 0]

        data['beta'] = betas

        # 性能优化：分离滚动均值和标准差计算
        data2 = data.dropna().copy().reset_index(drop=True)
        # 计算滚动均值
        data2['beta_mean'] = data2['beta'].rolling(M, min_periods=20).mean()
        # 计算滚动标准差
        data2['beta_std'] = data2['beta'].rolling(M, min_periods=20).std()
        # 避免除以0
        data2['beta_std'] = data2['beta_std'].replace(0, np.nan)
        # 计算标准分
        data2['std_score'] = (data2['beta'] - data2['beta_mean']) / data2['beta_std']

        # 生成交易信号
        data2['flag'] = 0
        data2['position'] = 0
        position = 0

        # 增加20日均线判断条件
        data2['ma20_condition'] = data2['ma20'].shift(1) > data2['ma20'].shift(3)

        for i in range(1, len(data2)):
            std_score = data2.loc[i, 'std_score']

            # 买入条件判断
            buy_condition = std_score > buy_thre
            if use_ma_filter:
                buy_condition = buy_condition and data2.loc[i, 'ma20_condition']

            if position == 0 and buy_condition:
                data2.loc[i, 'flag'] = 1
                data2.loc[i, 'position'] = 1  # 当日即建仓
                position = 1
            elif position == 1 and std_score < sell_thre:
                data2.loc[i, 'flag'] = -1
                data2.loc[i, 'position'] = 0  # 当日即平仓
                position = 0
            else:
                data2.loc[i, 'position'] = position  # 维持当前仓位

    except Exception as e:
        st.error(f"计算错误: {str(e)}")
    return data2, buy_thre, sell_thre


def calculate_strategy_performance(processed_data):
    """计算策略表现指标"""
    if processed_data.empty:
        return {}

    # 计算每日收益率
    processed_data['daily_return'] = processed_data['close'].pct_change()

    # 计算策略收益率（考虑信号）
    processed_data['strategy_return'] = processed_data['position'].shift(1) * processed_data['daily_return']

    # 计算累计收益率
    processed_data['cumulative_return'] = (1 + processed_data['strategy_return']).cumprod()

    # 计算基准累计收益率（买入持有）
    processed_data['benchmark_return'] = (1 + processed_data['daily_return']).cumprod()

    # 计算最大回撤
    processed_data['peak'] = processed_data['cumulative_return'].cummax()
    processed_data['drawdown'] = (processed_data['cumulative_return'] - processed_data['peak']) / processed_data['peak']
    max_drawdown = processed_data['drawdown'].min()

    # 计算总收益率
    total_return = processed_data['cumulative_return'].iloc[-1] - 1

    return {
        'total_return': total_return * 100,  # 转换为百分比
        'benchmark_return': (processed_data['benchmark_return'].iloc[-1] - 1) * 100,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': (processed_data['strategy_return'].mean() / processed_data['strategy_return'].std()) * np.sqrt(
            252)
    }


def main():
    st.markdown('<h1 style="color:#1E90FF; font-size:36px;">多指数RSRS指标分析系统</h1>', unsafe_allow_html=True)

    # === 参数自定义功能 ===
    st.sidebar.header("参数设置")
    N = st.sidebar.slider("滚动窗口大小 (N)", min_value=5, max_value=50, value=18, step=1)
    M = st.sidebar.slider("标准分计算窗口 (M)", min_value=100, max_value=1000, value=600, step=50)
    buy_thre = st.sidebar.slider("买入阈值", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    sell_thre = st.sidebar.slider("卖出阈值", min_value=-2.0, max_value=-0.1, value=-0.7, step=0.1)

    # 20日均线过滤开关
    use_ma_filter = st.sidebar.checkbox("增加20日均线判断买入", value=False)

    # 指数选择列表
    index_options = [
        "中证1000 (000852.SH)", "沪深300 (000300.SH)", "上证50 (000016.SH)",
        "上证指数 (000001.SH)", "北证50 (899050.BJ)",
        "科创综指 (000680.SH)", "恒生指数 (HSI.HK)", "恒生科技指数 (HSTECH.HK)"
    ]
    selected_index = st.selectbox("选择指数", index_options, index=0)
    index_name = re.search(r"(.+?)\s*\(", selected_index).group(1).strip()

    if st.button("计算指标", key="calculate_button"):
        with st.spinner(f"获取{index_name}数据中..."):
            data = fetch_index_data(index_name)
            if data.empty:
                st.error("数据获取失败，请检查网络或代码")
                return

            # 添加计算进度提示
            with st.spinner("计算指标中，请稍候..."):
                processed_data, buy_thre, sell_thre = calculate_beta_and_signals(
                    data, N, M, buy_thre, sell_thre, use_ma_filter)

            if processed_data.empty:
                st.warning("计算结果为空")
                return

            # 图表绘制
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
            title = f"{index_name} RSRS指标分析 (N={N}, M={M})"
            if use_ma_filter:
                title += " + 20日均线过滤"
            fig.suptitle(title, fontsize=20)

            # 价格与信号图 - 修改信号颜色
            ax1.plot(processed_data['trade_date'], processed_data['close'], 'b-', label='收盘价', linewidth=2)
            buy_points = processed_data[processed_data['flag'] == 1]
            sell_points = processed_data[processed_data['flag'] == -1]
            if not buy_points.empty:
                ax1.scatter(buy_points['trade_date'], buy_points['close'],
                            marker='^', color='r', s=120, label='买入')  # 红色向上三角形
            if not sell_points.empty:
                ax1.scatter(sell_points['trade_date'], sell_points['close'],
                            marker='v', color='g', s=120, label='卖出')  # 绿色向下三角形
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)

            # 标准分图
            ax2.plot(processed_data['trade_date'], processed_data['std_score'], 'purple', label='标准分')
            ax2.axhline(buy_thre, color='g', linestyle='--', label='买入阈值')
            ax2.axhline(sell_thre, color='r', linestyle='--', label='卖出阈值')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            plt.gcf().autofmt_xdate()
            st.pyplot(fig)

            # 最近数据表格
            st.subheader("最近十个交易日指标")
            recent_data = processed_data[['trade_date', 'close', 'beta', 'std_score', 'flag', 'position']].tail(10)
            recent_data = recent_data.rename(columns={
                'trade_date': '日期', 'close': '收盘价',
                'beta': 'Beta值', 'std_score': '标准分',
                'flag': '信号', 'position': '仓位'
            })
            recent_data['日期'] = recent_data['日期'].dt.strftime('%Y-%m-%d')

            # 信号状态映射
            def map_signal(row):
                if row['信号'] == 1:
                    return '买入'
                elif row['信号'] == -1:
                    return '卖出'
                elif row['仓位'] == 0:
                    return '空仓'
                else:
                    return '持有'

            recent_data['信号'] = recent_data.apply(map_signal, axis=1)

            # 表格样式
            numeric_cols = ['收盘价', 'Beta值', '标准分']
            for col in numeric_cols:
                recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')

            styled_table = recent_data.style.format(
                {col: '{:.4f}' for col in numeric_cols},
                na_rep="N/A"
            ).applymap(
                lambda x: 'color: red' if x == '买入' else (
                    'color: green' if x == '卖出' else (
                        'color: orange' if x == '空仓' else '')
                ), subset=['信号']
            )
            st.dataframe(styled_table, use_container_width=True, height=380)

            # === 策略表现统计 ===
            st.subheader("策略表现统计")
            col1, col2 = st.columns(2)

            # 计算信号统计
            buy_count = len(buy_points)
            sell_count = len(sell_points)

            col1.metric("买入信号次数", buy_count)
            col2.metric("卖出信号次数", sell_count)

            # === 回测结果 ===
            st.subheader("策略回测结果")
            performance = calculate_strategy_performance(processed_data)

            if performance:
                col3, col4, col5, col6 = st.columns(4)

                col3.metric("策略总收益率", f"{performance['total_return']:.2f}%")
                col4.metric("基准收益率", f"{performance['benchmark_return']:.2f}%")
                col5.metric("最大回撤", f"{performance['max_drawdown']:.2f}%")
                col6.metric("夏普比率", f"{performance['sharpe_ratio']:.2f}")

                # 绘制累计收益曲线
                fig2, ax = plt.subplots(figsize=(12, 6))
                ax.plot(processed_data['trade_date'], processed_data['cumulative_return'],
                        label=f'策略收益 ({performance["total_return"]:.2f}%)', color='blue')
                ax.plot(processed_data['trade_date'], processed_data['benchmark_return'],
                        label=f'基准收益 ({performance["benchmark_return"]:.2f}%)', color='orange', linestyle='--')
                ax.set_title('累计收益对比')
                ax.set_xlabel('日期')
                ax.set_ylabel('累计收益')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig2)
            else:
                st.warning("无法计算策略表现")


if __name__ == "__main__":
    main()