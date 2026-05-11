import requests
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

# 统计/数学库
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="澳洲幸运8 智能预测系统", page_icon="🎲", layout="wide")
st.title("🎰 澳洲幸运8 综合预测分析")
st.caption("📡 数据源：115kai3.com | 包含频率/遗漏/回归/ARIMA/马尔可夫/XGBoost/深度学习/易经/参伍依数 | 综合投票")

# ==================== 数据抓取 ====================
@st.cache_data(ttl=300)
def fetch_draws(limit=300):
    url = "https://115kai3.com/plan/api.do"
    params = {"code": "lucky8ball", "plan": "0", "_t": int(time.time()*1000)}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        openhis = data.get("data", {}).get("openhis", [])
        records = []
        for draw in openhis[:limit]:
            numbers_str = draw.get("openNums", [])
            if len(numbers_str) == 8:
                numbers = [int(x) for x in numbers_str]
                records.append({
                    "期号": draw.get("issue"),
                    "开奖时间": draw.get("time"),
                    "开奖号码": numbers,
                    "第一位": numbers[0]
                })
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

# ==================== 通用预测接口（返回下期第一位号码）====================
def freq_hot_cold(history_first, top_n=3):
    """频率分析：返回热号Top N 和冷号Bottom N"""
    cnt = Counter(history_first)
    hot = [num for num, _ in cnt.most_common(top_n)]
    cold = [num for num, _ in cnt.most_common()[:-top_n-1:-1]] if len(cnt) > top_n else hot
    return hot, cold

def freq_predict(history_first):
    """基于频率加权随机预测一个号码"""
    cnt = Counter(history_first)
    nums = list(cnt.keys())
    probs = np.array([cnt[n] for n in nums]) / len(history_first)
    return np.random.choice(nums, p=probs)

def missing_analysis(history_first):
    """遗漏分析：返回当前遗漏期数最大的号码（最冷）"""
    last_occurrence = {}
    for idx, num in enumerate(history_first):
        last_occurrence[num] = idx
    current_idx = len(history_first) - 1
    missing = {num: current_idx - last_idx for num, last_idx in last_occurrence.items()}
    # 遗漏最大的号码
    if missing:
        coldest = max(missing, key=missing.get)
        return coldest, missing
    return 1, {}

def regression_predict(history_first):
    """线性回归预测下期第一位（基于期数索引）"""
    X = np.arange(len(history_first)).reshape(-1, 1)
    y = np.array(history_first)
    model = LinearRegression()
    model.fit(X, y)
    next_idx = len(history_first)
    pred = model.predict([[next_idx]])[0]
    return int(np.clip(round(pred), 1, 20))

def arima_predict(history_first):
    """ARIMA时间序列预测"""
    try:
        model = ARIMA(history_first, order=(1,1,1))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=1)[0]
        return int(np.clip(round(pred), 1, 20))
    except:
        return np.random.randint(1, 21)

def markov_predict(history_first):
    """马尔可夫链转移概率预测下一期第一位"""
    trans = {}
    for i in range(len(history_first)-1):
        cur = history_first[i]
        nxt = history_first[i+1]
        trans.setdefault(cur, []).append(nxt)
    last = history_first[-1]
    if last in trans:
        next_vals = trans[last]
        cnt = Counter(next_vals)
        most = cnt.most_common(1)[0][0]
        return most
    else:
        return freq_predict(history_first)  # 回退

def xgboost_predict(history_first, window=10):
    """XGBoost：使用前window期特征预测下一期"""
    if len(history_first) < window+1:
        return freq_predict(history_first)
    X, y = [], []
    for i in range(len(history_first)-window):
        X.append(history_first[i:i+window])
        y.append(history_first[i+window])
    X = np.array(X)
    y = np.array(y)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    last_window = history_first[-window:]
    pred = model.predict([last_window])[0]
    return int(np.clip(round(pred), 1, 20))

def deep_learning_predict(history_first, window=10):
    """多层感知机 (MLP) 模拟深度学习"""
    if len(history_first) < window+1:
        return freq_predict(history_first)
    X, y = [], []
    for i in range(len(history_first)-window):
        X.append(history_first[i:i+window])
        y.append(history_first[i+window])
    X = np.array(X)
    y = np.array(y)
    model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    model.fit(X, y)
    last_window = history_first[-window:]
    pred = model.predict([last_window])[0]
    return int(np.clip(round(pred), 1, 20))

def yijing_predict():
    """易经预测（模拟：基于当前日期时间生成1-20数字）"""
    t = datetime.now()
    seed = t.year * 10000 + t.month * 100 + t.day
    np.random.seed(seed)
    return np.random.randint(1, 21)

def canwu_predict():
    """参伍依数（模拟：基于历史频率的某种变换）"""
    # 简单模拟：返回1-20中与当天星期相关的数字
    weekday = datetime.now().weekday()
    return (weekday % 20) + 1

def comprehensive_vote(predictions):
    """综合投票：取出现最多的数字，平局则取平均并四舍五入"""
    cnt = Counter(predictions)
    most_common = cnt.most_common()
    max_count = most_common[0][1]
    candidates = [num for num, count in most_common if count == max_count]
    if len(candidates) == 1:
        return candidates[0]
    else:
        return int(np.mean(candidates))

# ==================== Streamlit 界面 ====================
df = fetch_draws(limit=200)
if df.empty:
    st.stop()

first_history = df["第一位"].tolist()
latest = df.iloc[-1]

st.subheader("📌 最新开奖信息")
col1, col2 = st.columns(2)
with col1:
    st.metric("期号", latest["期号"])
with col2:
    st.metric("开奖时间", latest["开奖时间"])
st.write("**开奖号码：**", " ".join(f"{n:02d}" for n in latest["开奖号码"]))

# ==================== 分析结果 ====================
st.subheader("📊 历史数据统计")
# 频率分布图
cnt = Counter(first_history)
freq_df = pd.DataFrame(cnt.items(), columns=["号码", "次数"]).sort_values("号码")
fig, ax = plt.subplots()
ax.bar(freq_df["号码"], freq_df["次数"], color="skyblue")
ax.set_xlabel("第一位号码")
ax.set_ylabel("出现次数")
ax.set_title("第一位号码历史频次")
st.pyplot(fig)

# 热冷号
hot, cold = freq_hot_cold(first_history)
st.info(f"🔥 热号（高频）: {hot}   ❄️ 冷号（低频）: {cold}")

# 遗漏分析
coldest, missing_dict = missing_analysis(first_history)
st.info(f"📉 当前遗漏最久的号码: {coldest} (遗漏 {missing_dict[coldest]} 期)")

# ==================== 各模型预测 ====================
st.subheader("🔮 下期第一位号码预测（各种方法）")
with st.spinner("模型计算中，请稍候..."):
    pred_freq = freq_predict(first_history)
    pred_missing = coldest  # 遗漏最大者
    pred_reg = regression_predict(first_history)
    pred_arima = arima_predict(first_history)
    pred_markov = markov_predict(first_history)
    pred_xgb = xgboost_predict(first_history)
    pred_dl = deep_learning_predict(first_history)
    pred_yijing = yijing_predict()
    pred_canwu = canwu_predict()

    predictions = [pred_freq, pred_missing, pred_reg, pred_arima, pred_markov,
                   pred_xgb, pred_dl, pred_yijing, pred_canwu]
    pred_comp = comprehensive_vote(predictions)

# 显示预测结果表格
result_data = {
    "模型": ["频率分析", "遗漏分析", "线性回归", "ARIMA时间序列", "马尔可夫链",
             "XGBoost", "深度学习(MLP)", "易经预测", "参伍依数", "综合投票"],
    "预测第一位": [pred_freq, pred_missing, pred_reg, pred_arima, pred_markov,
                 pred_xgb, pred_dl, pred_yijing, pred_canwu, pred_comp]
}
result_df = pd.DataFrame(result_data)
st.dataframe(result_df, use_container_width=True)

# 雷达图或投票详情
st.subheader("🗳️ 综合投票详情")
vote_cnt = Counter(predictions)
vote_df = pd.DataFrame(vote_cnt.items(), columns=["号码", "得票"]).sort_values("得票", ascending=False)
st.bar_chart(vote_df.set_index("号码"))

st.success(f"🎯 最终综合预测号码： **{pred_comp}**")

# ==================== 历史回测（可选）====================
if st.checkbox("📈 显示最近10期预测回测（频率法）"):
    backtest = []
    for i in range(10, len(first_history)):
        train = first_history[:i]
        pred = freq_predict(train)
        actual = first_history[i]
        backtest.append({"实际": actual, "预测": pred, "命中": pred == actual})
    back_df = pd.DataFrame(backtest)
    st.dataframe(back_df.tail(20))
    accuracy = back_df["命中"].mean()
    st.metric("频率法回测准确率", f"{accuracy:.2%}")

st.caption("⚠️ 所有预测结果仅供娱乐研究，不构成投注建议。数据每5分钟自动刷新。")
