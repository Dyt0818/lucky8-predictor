import requests
import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="澳洲幸运8 玩法分析系统", page_icon="🎲", layout="wide")
st.title("🎰 澳洲幸运8 玩法全分析")
st.caption("📡 数据来源：115kai3.com | 1-20号码，8个开奖数字 | 展示番/角/车/念/无/堂/单双/大小/特码结果")

# ==================== 数据抓取 ====================
@st.cache_data(ttl=300)
def fetch_draws(limit=30):
    url = "https://115kai3.com/plan/api.do"
    params = {"code": "lucky8ball", "plan": "0", "_t": int(time.time()*1000)}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        openhis = data.get("data", {}).get("openhis", [])
        records = []
        for draw in openhis[:limit]:
            numbers_str = draw.get("openNums", [])
            if numbers_str and len(numbers_str) == 8:
                numbers = [int(x) for x in numbers_str]
                records.append({
                    "期号": draw.get("issue"),
                    "开奖时间": draw.get("time"),
                    "开奖号码": numbers
                })
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return pd.DataFrame()

# ==================== 玩法核心函数 ====================
def num_to_fan(num):
    """1,5,9,13,17→1 ; 2,6,10,14,18→2 ; 3,7,11,15,19→3 ; 4,8,12,16,20→4"""
    mapping = {1:1,5:1,9:1,13:1,17:1, 2:2,6:2,10:2,14:2,18:2, 3:3,7:3,11:3,15:3,19:3, 4:4,8:4,12:4,16:4,20:4}
    return mapping.get(num, None)

def check_fan(fan_num, numbers):
    """某番是否出现"""
    for n in numbers:
        if num_to_fan(n) == fan_num:
            return "赢"
    return "输"

def check_jiao(nums, numbers):
    """角：两个号码，出现任一即赢"""
    for b in nums:
        if b in numbers:
            return "赢"
    return "输"

def check_che(nums, numbers):
    """车（三门）：三个号码，出现任一即赢"""
    for b in nums:
        if b in numbers:
            return "赢"
    return "输"

def check_nian(win, he, numbers):
    """念：赢号出现=赢，和号出现=和，其他=输"""
    if win in numbers:
        return "赢"
    elif he in numbers:
        return "和"
    else:
        return "输"

def check_wu(win, he1, he2, numbers):
    """无：赢号=赢，和号1或和号2=和，其他=输"""
    if win in numbers:
        return "赢"
    elif he1 in numbers or he2 in numbers:
        return "和"
    else:
        return "输"

def check_tang(fan_num, numbers):
    """
    1正：开1番=赢，开2或4番=走水，开3番=输
    2正：开2番=赢，开1或3番=走水，开4番=输
    3正：开3番=赢，开2或4番=走水，开1番=输
    4正：开4番=赢，开1或3番=走水，开2番=输
    """
    fans = [num_to_fan(n) for n in numbers]
    if fan_num in fans:
        return "赢"
    if fan_num == 1:
        if 2 in fans or 4 in fans:
            return "走水"
        else:
            return "输"
    elif fan_num == 2:
        if 1 in fans or 3 in fans:
            return "走水"
        else:
            return "输"
    elif fan_num == 3:
        if 2 in fans or 4 in fans:
            return "走水"
        else:
            return "输"
    elif fan_num == 4:
        if 1 in fans or 3 in fans:
            return "走水"
        else:
            return "输"
    return "输"

def check_single_double(numbers):
    """单数个数>0则赢单，双数个数>0则赢双（通常为：有单即赢单）"""
    has_single = any(n % 2 == 1 for n in numbers)
    has_double = any(n % 2 == 0 for n in numbers)
    return ("赢" if has_single else "输"), ("赢" if has_double else "输")

def check_big_small(numbers):
    """大小：大于等于11为大"""
    has_big = any(n >= 11 for n in numbers)
    has_small = any(n <= 10 for n in numbers)
    return ("赢" if has_big else "输"), ("赢" if has_small else "输")

def check_tema(bet_num, pos, numbers):
    """特码：指定位置(pos从1开始)的号码等于bet_num则赢"""
    if 1 <= pos <= 8 and numbers[pos-1] == bet_num:
        return "赢"
    return "输"

# ==================== 生成所有玩法选项列表 ====================
all_play_items = []

# 番
for i in range(1,5):
    all_play_items.append(f"{i}番")

# 角（预定义一些常见角，可按需扩展）
jiao_combos = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]]
for c in jiao_combos:
    all_play_items.append(f"{c[0]}{c[1]}角")

# 车（三门示例：取1,2,3；4,5,6；7,8,9；10,11,12；13,14,15；16,17,18；19,20,1？ 简单取连续三个）
che_combos = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]
for c in che_combos:
    all_play_items.append(f"{c[0]}{c[1]}{c[2]}车")

# 念（严） 示例：1念2, 2念3, 3念4, 4念1
nian_items = [(1,2),(2,3),(3,4),(4,1)]
for w,h in nian_items:
    all_play_items.append(f"{w}念{h}")

# 无 示例：1无2, 2无3, 3无4, 4无1
wu_items = [(1,2,3),(2,3,4),(3,4,1),(4,1,2)]  # (赢,和1,和2)
for w,h1,h2 in wu_items:
    all_play_items.append(f"{w}无{h1}{h2}")

# 堂（正）
for i in range(1,5):
    all_play_items.append(f"{i}正")

# 单双
all_play_items.append("单")
all_play_items.append("双")

# 大小
all_play_items.append("大")
all_play_items.append("小")

# 特码（举例每个位置投注1号，可自行调整。这里展示位置1的特码）
for pos in range(1,9):
    all_play_items.append(f"特码{pos}位=01")

# ==================== 对每一期计算所有玩法结果 ====================
def compute_results_for_draw(numbers):
    """返回一个字典，键=玩法名称，值=结果字符串"""
    res = {}
    # 番
    for i in range(1,5):
        res[f"{i}番"] = check_fan(i, numbers)
    # 角
    for c in jiao_combos:
        res[f"{c[0]}{c[1]}角"] = check_jiao(c, numbers)
    # 车
    for c in che_combos:
        res[f"{c[0]}{c[1]}{c[2]}车"] = check_che(c, numbers)
    # 念
    for w,h in nian_items:
        res[f"{w}念{h}"] = check_nian(w, h, numbers)
    # 无
    for w,h1,h2 in wu_items:
        res[f"{w}无{h1}{h2}"] = check_wu(w, h1, h2, numbers)
    # 堂
    for i in range(1,5):
        res[f"{i}正"] = check_tang(i, numbers)
    # 单双
    s, d = check_single_double(numbers)
    res["单"] = s
    res["双"] = d
    # 大小
    b, sm = check_big_small(numbers)
    res["大"] = b
    res["小"] = sm
    # 特码（固定每个位置投注01，可改为任意）
    for pos in range(1,9):
        res[f"特码{pos}位=01"] = check_tema(1, pos, numbers)
    return res

# ==================== 主界面 ====================
df = fetch_draws(limit=20)
if df.empty:
    st.stop()

# 展示最新一期号码
latest = df.iloc[-1]
st.subheader("📌 最新开奖信息")
col1, col2 = st.columns(2)
with col1:
    st.metric("期号", latest["期号"])
with col2:
    st.metric("开奖时间", latest["开奖时间"])
st.write("**开奖号码：**", " ".join(f"{n:02d}" for n in latest["开奖号码"]))

# 为每一期计算结果
results_per_draw = {}
for idx, row in df.iterrows():
    results_per_draw[row["期号"]] = compute_results_for_draw(row["开奖号码"])

# 构建玩法-期号矩阵
all_plays = all_play_items
draw_ids = df["期号"].tolist()

# 创建DataFrame，行=玩法，列=期号
matrix_data = {}
for draw_id in draw_ids:
    matrix_data[draw_id] = [results_per_draw[draw_id].get(play, "") for play in all_plays]

result_df = pd.DataFrame(matrix_data, index=all_plays)
# 转置以便阅读（期号为列，玩法为行，通常更喜欢期号在上方）
# 其实保持现在这样：行是玩法，列是期号，更方便横向比较
st.subheader("📊 各期玩法结果一览")
st.dataframe(result_df, use_container_width=True, height=600)

# 针对最新一期，单独列出详细结果
st.subheader("🔍 最新一期玩法详细结果")
latest_play_results = results_per_draw[latest["期号"]]
latest_df = pd.DataFrame(list(latest_play_results.items()), columns=["玩法", "结果"])
st.dataframe(latest_df, use_container_width=True)

st.caption("✅ 展示所有预定义玩法（番、角、车、念、无、堂、单双、大小、特码）在最近500期的输赢结果。数据每5分钟自动刷新。")
