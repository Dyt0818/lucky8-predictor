import requests
import pandas as pd
import streamlit as st
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ====== 配置 ======
DATA_API = "https://115kai3.com/plan/api.do"
PARAMS = {"code": "lucky8ball", "plan": "0"}
HISTORY_BACKUP = "lucky8_history.csv"

# ====== 1. 数据抓取（使用稳定接口）======
@st.cache_data(ttl=300)
def fetch_draws(limit=100):
    try:
        resp = requests.get(DATA_API, params={**PARAMS, "_t": int(time.time()*1000)}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        openhis = data.get("data", {}).get("openhis", [])
        records = []
        for draw in openhis[:limit]:
            draw_id = draw.get("issue")
            draw_time = draw.get("time")
            numbers_str = draw.get("openNums", [])
            if numbers_str and len(numbers_str) == 8:
                numbers = [int(x) for x in numbers_str]
                first_ball = numbers[0]
                records.append({
                    "draw_id": draw_id,
                    "draw_time": draw_time,
                    "first_ball": first_ball,
                    "numbers": numbers
                })
        df = pd.DataFrame(records)
        # 本地备份
        if not df.empty:
            df.to_csv(HISTORY_BACKUP, index=False)
        return df
    except Exception as e:
        st.error(f"实时数据获取失败: {e}")
        # 降级读取本地备份
        try:
            df = pd.read_csv(HISTORY_BACKUP)
            st.warning("使用本地缓存数据")
            return df
        except:
            return pd.DataFrame()

# ====== 2. 预测算法 ======
def freq_predictor(history_firsts, window=20):
    if len(history_firsts) == 0:
        return None, pd.Series()
    recent = history_firsts[-window:] if len(history_firsts) >= window else history_firsts
    counts = pd.Series(recent).value_counts().sort_index()
    if counts.sum() == 0:
        return None, counts
    probs = counts / counts.sum()
    predicted = np.random.choice(probs.index, p=probs.values)
    return predicted, counts

def markov_predictor(history_firsts):
    if len(history_firsts) < 2:
        return None
    last = history_firsts[-1]
    trans = {}
    for i in range(len(history_firsts)-1):
        cur = history_firsts[i]
        nxt = history_firsts[i+1]
        trans.setdefault(cur, []).append(nxt)
    if last not in trans:
        return None
    cnt = Counter(trans[last])
    return cnt.most_common(1)[0][0]

# ====== 3. Streamlit UI ======
st.set_page_config(page_title="澳洲幸运8 预测系统", page_icon="🎲", layout="wide")
st.title("🎰 澳洲幸运8 Lucky Ball 8 实时预测系统")
st.caption("📡 数据源: 115kai3.com | 仅供娱乐与统计分析")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 参数")
    window = st.slider("分析最近多少期", 10, 100, 20, step=5)
    auto_refresh = st.checkbox("自动刷新 (每5分钟)")
    if st.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()

# 自动刷新（需要安装 streamlit-autorefresh，可选）
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=300000, key="refresh")
    except ImportError:
        st.info("如需自动刷新，请运行: pip install streamlit-autorefresh")

# 加载数据
df = fetch_draws(limit=200)
if df.empty:
    st.stop()

first_history = df["first_ball"].tolist()

# 最近10期显示
st.subheader("📊 最近开奖记录 (最新10期)")
show_df = df[["draw_id","draw_time","first_ball","numbers"]].tail(10).copy()
show_df["numbers"] = show_df["numbers"].apply(lambda x: ", ".join(map(str, x)))
show_df.columns = ["期号","时间","第一位","全部号码"]
st.dataframe(show_df, use_container_width=True)

# 频率图表
st.subheader(f"📈 第一位号码频率 (最近{window}期)")
_, counts = freq_predictor(first_history, window)
if not counts.empty:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color="orange")
        ax.set_xlabel("号码")
        ax.set_ylabel("次数")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        st.pyplot(fig2)

# 预测
st.subheader("🔮 下一期第一位预测")
pred_freq, _ = freq_predictor(first_history, window)
pred_markov = markov_predictor(first_history)
pred_random = np.random.randint(1, 21)  # 澳洲幸运8号码范围 1~20

c1, c2, c3 = st.columns(3)
c1.metric("🎲 频率加权", pred_freq if pred_freq else "N/A")
c2.metric("📈 马尔可夫链", pred_markov if pred_markov else "N/A")
c3.metric("🔀 纯随机 (1-20)", pred_random)

# 回测
if len(first_history) >= 2:
    last_actual = first_history[-1]
    if len(first_history) >= window + 1:
        last_pred, _ = freq_predictor(first_history[:-1], window)
        if last_pred is not None:
            st.sidebar.success(f"上一期预测: {last_pred} → 实际: {last_actual} " +
                               ("✅ 命中" if last_pred == last_actual else "❌ 未命中"))
    else:
        st.sidebar.info("数据不足，无法回测")

st.caption("⚠️ 本系统仅为统计娱乐，不保证中奖。理性投注。")
