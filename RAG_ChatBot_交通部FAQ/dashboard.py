import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import io

# --- 頁面設定 ---
st.set_page_config(page_title="交通部 AI 問答後台", layout="wide")

# --- 從 SQLite 載入資料 ---
@st.cache_data

def load_data():
    conn = sqlite3.connect("feedback.db")
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df

df = load_data()

# --- 側邊選單 ---
menu = st.sidebar.radio("選擇頁面", [
    "📊 儀表板首頁",
    "📁 回饋紀錄查詢",
    "🔥 熱點問題分析",
    "📤 資料匯出"
])

# --- 1️⃣ 儀表板首頁 ---
if menu == "📊 儀表板首頁":
    st.title("📊 儀表板總覽")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("主題分類統計")
        st.bar_chart(df["topic"].value_counts())

    with col2:
        st.subheader("滿意度分佈")
        st.bar_chart(df["helpful"].value_counts())

    st.subheader("資料總筆數")
    st.metric("回饋數量", len(df))

# --- 2️⃣ 回饋紀錄查詢 ---
elif menu == "📁 回饋紀錄查詢":
    st.title("📁 問題與回饋資料表")

    with st.expander("🔍 篩選條件", expanded=True):
        category = st.multiselect("選擇主題分類", options=df["topic"].unique(), default=list(df["topic"].unique()))
        keyword = st.text_input("輸入關鍵字搜尋（可搜尋問題或回答）")

    filtered_df = df[df["topic"].isin(category)]
    if keyword:
        filtered_df = filtered_df[filtered_df["question"].str.contains(keyword, na=False) | filtered_df["answer"].str.contains(keyword, na=False)]

    st.dataframe(filtered_df, use_container_width=True)

# --- 3️⃣ 熱點問題分析 ---
elif menu == "🔥 熱點問題分析":
    st.title("🔥 熱門主題分析")

    st.subheader("不滿意比例最高的主題")
    if len(df) > 0:
        unsatisfied_df = df[df["helpful"] == "否"]
        rate_df = unsatisfied_df["topic"].value_counts() / df["topic"].value_counts()
        st.bar_chart(rate_df.fillna(0))

    st.subheader("有錯誤補充的紀錄")
    with st.expander("點擊查看"):
        st.dataframe(df[df["report"].notna() & (df["report"] != "")], use_container_width=True)

# --- 4️⃣ 資料匯出 ---
elif menu == "📤 資料匯出":
    st.title("📤 資料匯出工具")
    st.markdown("可自訂篩選條件並匯出回饋記錄為 CSV。")

    category = st.multiselect("選擇主題分類", options=df["topic"].unique(), default=list(df["topic"].unique()))
    filtered_export = df[df["topic"].isin(category)]

    st.download_button(
        label="📥 下載 CSV",
        data=filtered_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="feedback_export.csv",
        mime="text/csv"
    )