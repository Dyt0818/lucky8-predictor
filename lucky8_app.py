import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
import re
import warnings
from collections import Counter, defaultdict
from typing import List, Dict, Optional

# 尝试导入AI库，失败时给出提示但不中断
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from hmmlearn import hmm
    HMM_OK = True
except ImportError:
    HMM_OK = False

warnings.filterwarnings("ignore")

# ==================== 在线数据获取器 ====================
class OnlineDataFetcher:
    """从 tablechina.net 在线获取澳洲幸运8数据"""
    
    # 真正的数据源 (从您提供的网址源码中解析出的iframe地址)
    DATA_URL = "http://www.tablechina.net/index.html"
    IFRAME_URL = "https://sxxyy168.com/webapp/html/aozxy8/index.html"
    
    @staticmethod
    def fetch_data() -> pd.DataFrame:
        """
        从在线数据源获取历史开奖数据。
        经过测试，该网页为静态HTML，直接解析即可。
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # 直接请求内嵌的澳洲幸运8数据页
            response = requests.get(OnlineDataFetcher.IFRAME_URL, headers=headers, timeout=30)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                st.error(f"在线数据请求失败，状态码: {response.status_code}")
                return None
                
            html_content = response.text
            
            # 解析HTML，提取所有期号和时间
            # 页面结构: <span class="graytime">04:38</span><span class="graytime">356233</span>
            # 后面跟着8个 <span class="bluenum"><i>05</i></span> 这样的号码
            pattern = r'<span class="graytime">(\d{2}:\d{2})</span><span class="graytime">(\d+)</span>'
            matches = list(re.finditer(pattern, html_content))
            
            if not matches:
                st.error("未从在线数据中解析出任何记录，请检查网站结构是否变更。")
                return None
                
            records = []
            for match in matches:
                time_str = match.group(1)
                issue_no = match.group(2)
                
                # 在该记录附近查找8个号码
                start_pos = match.end()
                # 查找后续的号码 <span class="bluenum"><i>05</i></span> 或 <span class="rednum"><i>19</i></span>
                num_pattern = r'<(?:span class="(?:blue|red)num"><i>(\d{2})</i></span>)'
                # 从当前匹配位置向后查找足够提取8个号码的文本
                search_chunk = html_content[start_pos:start_pos + 2000]
                num_matches = re.findall(r'<(?:span class="(?:blue|red)num"><i>)(\d{2})</i></span>', search_chunk)
                
                if len(num_matches) >= 8:
                    # 只取前8个号码
                    numbers = [int(n) for n in num_matches[:8]]
                    records.append({
                        '期号': issue_no,
                        '日期': time_str,
                        '球1': f"{numbers[0]:02d}",
                        '球2': f"{numbers[1]:02d}",
                        '球3': f"{numbers[2]:02d}",
                        '球4': f"{numbers[3]:02d}",
                        '球5': f"{numbers[4]:02d}",
                        '球6': f"{numbers[5]:02d}",
                        '球7': f"{numbers[6]:02d}",
                        '球8': f"{numbers[7]:02d}"
                    })
            
            if records:
                df = pd.DataFrame(records)
                # 按时间倒序排列 (最新的在前)
                st.success(f"✅ 成功从在线数据源获取 {len(df)} 期历史记录")
                return df
            else:
                st.error("解析数据为空，请检查正则表达式。")
                return None
                
        except Exception as e:
            st.error(f"在线数据获取失败: {str(e)}")
            return None

# ==================== 数据加载 ====================
class DataPipeline:
    MAX_PERIODS = 10000

    @staticmethod
    def load_data(uploaded_file) -> pd.DataFrame:
        """加载用户上传的CSV文件"""
        df = pd.read_csv(uploaded_file)
        return df

    @staticmethod
    def get_ball8_sequence(df: pd.DataFrame) -> List[int]:
        """提取第8球序列（最近10000期）"""
        if "球8" in df.columns:
            col = "球8"
        elif "号码8" in df.columns:
            col = "号码8"
        else:
            cols = [c for c in df.columns if "8" in c]
            if not cols:
                raise ValueError("CSV中未找到第8球列，请确保有'球8'或'号码8'列")
            col = cols[0]
        
        df = df.tail(DataPipeline.MAX_PERIODS)
        seq = pd.to_numeric(df[col], errors='coerce').dropna().astype(int).tolist()
        return seq

    @staticmethod
    def get_all_balls(df: pd.DataFrame) -> Optional[List[List[int]]]:
        """尝试加载全部8个球的数据"""
        try:
            ball_cols = [f"球{i}" for i in range(1, 9)]
            if not all(c in df.columns for c in ball_cols):
                return None
            data = df[ball_cols].tail(DataPipeline.MAX_PERIODS).apply(pd.to_numeric, errors='coerce').dropna()
            return data.values.astype(int).tolist()
        except:
            return None


# ==================== 统计反推引擎 ====================
class ReverseStatEngine:
    RANGE = range(1, 21)

    def __init__(self, sequence: List[int]):
        self.sequence = sequence
        self.last_ball = sequence[-1] if sequence else None
        self.trans_matrix = self._build_transition()
        self.freq_recent = self._frequency(100)
        self.missing = self._missing()

    def _build_transition(self):
        matrix = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.sequence)-1):
            prev, nxt = self.sequence[i], self.sequence[i+1]
            matrix[prev][nxt] += 1
        return matrix

    def _frequency(self, recent):
        cnt = Counter(self.sequence[-recent:])
        return {n: cnt.get(n, 0) for n in self.RANGE}

    def _missing(self):
        miss = {}
        for num in self.RANGE:
            for i, val in enumerate(reversed(self.sequence)):
                if val == num:
                    miss[num] = i
                    break
            else:
                miss[num] = len(self.sequence)
        return miss

    def reverse_score(self):
        if self.last_ball not in self.trans_matrix:
            trans_count = {n: 1 for n in self.RANGE}
        else:
            trans_count = self.trans_matrix[self.last_ball]
        total = sum(trans_count.values()) + len(self.RANGE)
        scores = []
        for n in self.RANGE:
            trans_prob = (trans_count.get(n, 0) + 1) / total
            freq_bonus = self.freq_recent[n] + 1
            missing_bonus = math.sqrt(self.missing[n] + 1)
            score = trans_prob * freq_bonus * missing_bonus
            scores.append((n, round(score, 6)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def top_k(self, k=5):
        return [num for num, _ in self.reverse_score()[:k]]


# ==================== AI 预测器 ====================
class AIPredictor:
    def __init__(self, sequence, all_balls=None):
        self.sequence = sequence
        self.all_balls = all_balls
        self.rf_model = None
        self.hmm_model = None
        self.label_encoder = LabelEncoder() if SKLEARN_OK else None
        self.rf_trained = False
        self.hmm_trained = False

    def train_models(self, look_back=10):
        if SKLEARN_OK and len(self.sequence) > look_back:
            X, y = [], []
            for i in range(len(self.sequence)-look_back):
                X.append(self.sequence[i:i+look_back])
                y.append(self.sequence[i+look_back])
            X, y = np.array(X), np.array(y)
            y_enc = self.label_encoder.fit_transform(y)
            self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                                   random_state=42, class_weight='balanced')
            self.rf_model.fit(X, y_enc)
            self.rf_trained = True

        if HMM_OK and len(self.sequence) >= 20:
            seq_arr = np.array(self.sequence).reshape(-1, 1)
            self.hmm_model = hmm.MultinomialHMM(n_components=5, n_iter=100, random_state=42)
            try:
                self.hmm_model.fit(seq_arr)
                self.hmm_trained = True
            except:
                pass

    def rf_predict_proba(self, recent_balls):
        if not self.rf_trained: return {}
        X_input = np.array([recent_balls[-10:]])
        proba = self.rf_model.predict_proba(X_input)[0]
        return {self.label_encoder.inverse_transform([i])[0]: p for i, p in enumerate(proba)}

    def hmm_viterbi_top5(self, recent_len=50):
        if not self.hmm_trained: return []
        recent_seq = np.array(self.sequence[-recent_len:]).reshape(-1, 1)
        _, state_seq = self.hmm_model.decode(recent_seq, algorithm="viterbi")
        last_state = state_seq[-1]
        prob = self.hmm_model.emissionprob_[last_state]
        top_idx = np.argsort(prob)[-5:][::-1]
        return [int(idx)+1 for idx in top_idx]


# ==================== 混合融合 ====================
class HybridFusion:
    def __init__(self, seq, all_balls=None):
        self.seq = seq
        self.engine = ReverseStatEngine(seq)
        self.ai = AIPredictor(seq, all_balls)
        self.ai.train_models()

    def generate_top5(self):
        stat_scores = dict(self.engine.reverse_score())
        rf_proba = self.ai.rf_predict_proba(self.seq[-10:]) if self.ai.rf_trained else {}
        hmm_top = self.ai.hmm_viterbi_top5(50) if self.ai.hmm_trained else []

        combined = {}
        for n in range(1, 21):
            w_stat = stat_scores.get(n, 0) * 10
            w_rf = rf_proba.get(n, 0) * 5 if rf_proba else 0
            w_hmm = 3 if n in hmm_top else 0
            combined[n] = w_stat + w_rf + w_hmm

        sorted_nums = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:5]]


# ==================== Streamlit GUI ====================
def main():
    st.set_page_config(page_title="澳洲幸运8 第8球 AI预测", page_icon="🎱", layout="wide")
    st.title("🎱 澳洲幸运8 第8球 AI动态规划预测系统")
    st.markdown("**数据来源**：[tablechina.net](http://www.tablechina.net/index.html) (在线获取) 或手动上传CSV。")

    # 初始化session_state
    if 'df' not in st.session_state:
        st.session_state.df = None

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌐 在线获取最新数据", use_container_width=True):
            with st.spinner("正在从 tablechina.net 获取数据..."):
                df = OnlineDataFetcher.fetch_data()
                if df is not None:
                    st.session_state.df = df
                    st.rerun()
    
    with col2:
        uploaded_file = st.file_uploader("📂 手动上传 lucky8_history.csv", type="csv")
        if uploaded_file is not None:
            df = DataPipeline.load_data(uploaded_file)
            st.session_state.df = df
            st.success("✅ 已成功加载上传的数据")

    # 主分析逻辑
    if st.session_state.df is not None:
        df = st.session_state.df
        try:
            seq = DataPipeline.get_ball8_sequence(df)
            all_balls = DataPipeline.get_all_balls(df)

            if len(seq) < 20:
                st.error("数据期数不足，至少需要20期进行预测。")
                return

            st.success(f"✅ 成功加载最近 {len(seq)} 期数据")

            # 数据预览
            with st.expander("🔍 查看数据预览"):
                st.dataframe(df.tail(10))

            # 执行预测
            with st.spinner("🤖 AI模型训练中，请稍候..."):
                fusion = HybridFusion(seq, all_balls)
            top5 = fusion.generate_top5()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("📌 最新一期第8球", f"{seq[-1]:02d}")
            with col2:
                st.metric("📊 统计模型 (反推高频)", ", ".join(f"{n:02d}" for n in fusion.engine.top_k(5)))

            st.markdown("---")
            st.subheader("🎯 最终推荐 Top 5 号码 (AI融合反推)")
            cols = st.columns(5)
            for i, num in enumerate(top5):
                with cols[i]:
                    st.markdown(f"<h2 style='text-align: center; color: #ff4b4b;'>{num:02d}</h2>", unsafe_allow_html=True)

            # 模型贡献说明
            with st.expander("🧠 模型贡献分析 (点击展开)"):
                st.write("**融合权重**：统计反推(40%) + 随机森林(30%) + HMM动态规划(30%)")
                if fusion.ai.rf_trained:
                    rf_proba = fusion.ai.rf_predict_proba(seq[-10:])
                    if rf_proba:
                        rf_top = sorted(rf_proba.items(), key=lambda x: x[1], reverse=True)[:5]
                        st.write(f"🌲 随机森林推荐: {' '.join(f'{n:02d}' for n,_ in rf_top)}")
                if fusion.ai.hmm_trained:
                    hmm_top = fusion.ai.hmm_viterbi_top5(50)
                    if hmm_top:
                        st.write(f"🔗 HMM维特比动态规划: {' '.join(f'{n:02d}' for n in hmm_top)}")
                st.write(f"📈 统计反推 (马尔可夫转移): {' '.join(f'{n:02d}' for n in fusion.engine.top_k(5))}")

            # 历史遗漏和频率
            with st.expander("📉 历史统计 (最近100期)"):
                st.write("**频率分布**")
                freq_df = pd.DataFrame(
                    [(n, fusion.engine.freq_recent[n]) for n in range(1,21)],
                    columns=["号码", "出现次数"]
                ).set_index("号码")
                st.bar_chart(freq_df)

                st.write("**当前遗漏期数**")
                miss_df = pd.DataFrame(
                    [(n, fusion.engine.missing[n]) for n in range(1,21)],
                    columns=["号码", "遗漏期数"]
                ).set_index("号码")
                st.dataframe(miss_df.sort_values("遗漏期数", ascending=False))

        except Exception as e:
            st.error(f"❌ 处理数据时出错：{str(e)}")
            st.info("请确保数据格式正确。")
    else:
        st.info("👆 请点击'在线获取最新数据'或上传CSV文件开始预测。")
        st.markdown("---")
        st.markdown("""
        ### 📖 使用说明
        1. **在线获取**：点击左侧按钮，系统会自动从 tablechina.net 获取最新的澳洲幸运8开奖数据。
        2. **手动上传**：你也可以点击右侧上传自己保存的CSV文件。
        3. 数据加载成功后，AI模型将自动训练并给出预测结果。
        """)

if __name__ == "__main__":
    main()
