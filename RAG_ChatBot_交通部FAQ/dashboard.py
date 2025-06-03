import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import io
from datetime import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="交通部 AI 後台系統", layout="wide")

# --- 自訂主題色 ---
custom_css = """
<style>
body {
    background-color: #0f1c2e;
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #16243d;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
button, .stButton>button {
    background-color: #264e86;
    color: white;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 從 SQLite 載入資料 ---
@st.cache_data
def load_data():
    conn = sqlite3.connect("feedback.db")
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df

df = load_data()
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# --- 側邊選單 ---
menu = st.sidebar.radio("導航選單", [
    "📊 摘要報告",
    "📁 資料總覽",
    "📤 資料匯出",
    "🔍 主題搜尋"
])

# --- 摘要報告 ---
if menu == "📊 摘要報告":
    st.title("📊 摘要報告")
    start_date = st.date_input("起始日期", value=df["time"].min().date())
    end_date = st.date_input("結束日期", value=df["time"].max().date())

    mask = (df["time"].dt.date >= start_date) & (df["time"].dt.date <= end_date)
    filtered = df[mask]

    st.subheader(f"資料總筆數：{len(filtered)}")

    topic_counts = filtered["topic"].replace({
        "交通違規": "Violation",
        "大眾運輸": "Public Transport",
        "道路建設": "Infrastructure",
        "政策建議": "Policy Advice",
        "其他": "Other"
    }).value_counts()
    topic_percent = (topic_counts / len(filtered) * 100).round(1)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(topic_counts.index, topic_counts.values, color="#264e86")
    for bar, pct in zip(bars, topic_percent):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"{int(height)}\n({pct}%)", ha='center', va='bottom', color='white')
    ax.set_ylabel("Count")
    ax.set_title("Topic Distribution", color="white")
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#0f1c2e')
    ax.set_facecolor('#0f1c2e')
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Arial'
    st.pyplot(fig)

    st.subheader("主題關鍵字統計：")
    for topic in filtered["topic"].unique():
        keywords = filtered[filtered["topic"] == topic]["question"].str.cat(sep=" ")
        keyword_list = pd.Series(keywords.split()).value_counts().head(5).index.tolist()
        st.markdown(f"**{topic}**：{', '.join(keyword_list)}")

# --- 資料總覽 ---
elif menu == "📁 資料總覽":
    st.title("📁 民意資料瀏覽")
    topic_labels = ["交通違規", "大眾運輸", "道路建設", "政策建議", "其他"]
    tabs = st.tabs(topic_labels)

    for i, tab in enumerate(tabs):
        with tab:
            topic_key = topic_labels[i]
            st.subheader(topic_key)
            topic_df = df[df["topic"] == topic_key]  # 精準比對
            keyword = st.text_input("🔍 搜尋此分類內容", key=f"kw_{i}")
            if keyword:
                topic_df = topic_df[
                    topic_df["question"].str.contains(keyword, case=False, na=False) |
                    topic_df["answer"].str.contains(keyword, case=False, na=False)
                ]
            for _, row in topic_df.iterrows():
                with st.container():
                    st.markdown(f"**問題摘要：** {row['question'][:60]}...")
                    st.markdown(f"📌 分類：{row['topic']} ｜ 🕒 時間：{row['time'].strftime('%Y/%m/%d %H:%M')}  ")
                    if 'answer' in row and row['answer']:
                        st.markdown(f"💬 回覆內容：{row['answer'][:100]}...")
                    if 'helpful' in row:
                        st.markdown(f"👍 滿意度：{'滿意' if row['helpful']=='是' else '不滿意'}")
                    if 'report' in row and row['report']:
                        st.markdown(f"📝 補充說明：{row['report']}")
                    st.markdown("---")

# --- 資料匯出 ---
elif menu == "📤 資料匯出":
    st.title("📤 資料匯出工具")
    category = st.multiselect("主題分類", options=df["topic"].unique(), default=list(df["topic"].unique()))
    start_date = st.date_input("起始日期", value=df["time"].min().date(), key="export_start")
    end_date = st.date_input("結束日期", value=df["time"].max().date(), key="export_end")
    export_df = df[df["topic"].isin(category)]
    export_df = export_df[(export_df["time"].dt.date >= start_date) & (export_df["time"].dt.date <= end_date)]

    st.download_button("⬇️ 匯出 CSV", export_df.to_csv(index=False).encode("utf-8-sig"), file_name="export.csv", mime="text/csv")
    st.download_button("⬇️ 匯出 JSON", export_df.to_json(orient="records", force_ascii=False), file_name="export.json", mime="application/json")

# --- 主題搜尋 ---
elif menu == "🔍 主題搜尋":
    st.title("🔍 主題快速搜尋")
    search_kw = st.text_input("請輸入欲查詢的關鍵字")
    if search_kw:
        result_df = df[df["question"].str.contains(search_kw, case=False, na=False)]
        st.write(f"共找到 {len(result_df)} 筆資料：")
        st.dataframe(result_df[["time", "topic", "question", "answer"]], use_container_width=True)
