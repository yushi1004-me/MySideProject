import pandas as pd
import numpy as np
import faiss
import gradio as gr
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
import sqlite3
import csv
import datetime
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizers錯誤

# 初始化與載入
df = pd.read_csv("faq_data.csv")  # FAQ的原始資料（問題、答案）
with open("faq_texts.pkl", "rb") as f:
    faq_texts = pickle.load(f)  # 結合過的文字語料，用於查詢比對
index = faiss.read_index("faq.index")  # FAISS 建立的語意查詢資料庫
model = SentenceTransformer("BAAI/bge-large-zh")  # hugguing face上的將文字轉向量的中文語意模型
# 初始化 Groq API
client = Groq(api_key="*********")  # 用來發送LLM請求（Groq API）

# 初始化 SQLite 資料庫（如不存在則創建）
def init_db():
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        question TEXT,
        answer TEXT,
        helpful TEXT,
        report TEXT,
        topic TEXT
    )''')
    conn.commit()
    conn.close()

init_db()


# 嵌入與回答
def embed(texts):
    return model.encode(
        [f"為這句話生成表示以用於檢索: {t}" for t in texts],
        normalize_embeddings=True
    )

def rephrase_answer(question, original_answer, chat_history):
    system_prompt = (
    "你是交通部的 AI 語言助理，請使用繁體中文，以自然、親切、專業的方式回覆民眾。"
    "請根據下方提供的原始回答進行重寫，使其更易懂、更自然。"
    "在回答中，如果與民眾意見反映或建議有關，請引導他們前往部長信箱：https://poms.motc.gov.tw/Daoan/tw。"
    "所有回答必須使用繁體中文，不得出現『使用者：』或『AI：』等格式。"
    "如果你無法回答問題，請說："
    "「對不起，您問的問題我目前無法回答，詳情請洽交通部客服專線詢問：0800-231-161。」"
    )


    context = "。".join([q for q, _ in chat_history[-2:]])
    user_prompt = (
        f"問題背景：{context}\n\n"
        f"原始回答：{original_answer}\n\n"
        f"請重新表達這段內容，使其自然易懂。"
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()

# LLM問題自動分類
def classify_topic_with_llm(question):
    system_prompt = (
        "你是交通部的內部客服資料分類員，請協助將民眾的問題依照主題分類。"
        "請從下列五個主題中選擇最符合此問題的分類："
        "交通違規、大眾運輸、道路建設、政策建議、其他。"
        "請僅回覆這五個類別中的一個，不要加上多餘說明。"
    )
    user_prompt = f"問題內容：{question}\n請回覆對應主題："

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()


# 查詢流程
def answer_question(user_input, chat_history):
    query_vector = embed([user_input])[0].astype("float32").reshape(1, -1)
    D, I = index.search(query_vector, k=1)
    original_answer = df.iloc[I[0][0]]['答覆']
    rewritten = rephrase_answer(user_input, original_answer, chat_history)
    chat_history.append((user_input, rewritten))
    return chat_history, chat_history, rewritten, ""

# 問答回饋紀錄
def record_feedback(chat_history, helpful, report_text):
    timestamp = datetime.datetime.now().isoformat()
    last_q, last_a = chat_history[-1]
    topic = classify_topic_with_llm(last_q)

    # 寫入 TXT（可選）
    log = (
        f"[{timestamp}]\n"
        f"Q: {last_q}\n"
        f"A: {last_a}\n"
        f"👍Helpful: {helpful}\n"
        f"🐞Report: {report_text}\n\n"
    )
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(log)

    # 寫入 CSV
    csv_file = Path("feedback_log.csv")
    file_exists = csv_file.exists()
    with open(csv_file, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["時間", "問題", "回答", "滿意與否", "錯誤補充", "主題分類"])
        writer.writerow([timestamp, last_q, last_a, helpful, report_text, topic])

    # 寫入 SQLite
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (time, question, answer, helpful, report, topic)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, last_q, last_a, helpful, report_text, topic))
    conn.commit()
    conn.close()

    # 回傳空歷史 → 重設多輪對話
    return [], [], "✅ 感謝您的回饋，我們已記錄。"


# UI 建構
with gr.Blocks() as demo:
    gr.Markdown("## 交通部 AI 問答助理（多輪對話 + 回饋機制）")

    chat_state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(lines=2, label="請輸入您的問題")
        submit_btn = gr.Button("送出問題")

    chat_display = gr.Chatbot(label="對話紀錄")
    current_answer = gr.Textbox(label="目前回答", interactive=False)

    with gr.Row():
        good_btn = gr.Button("滿意")
        bad_btn = gr.Button("不滿意")

    report_input = gr.Textbox(label="如有錯誤，可補充說明", lines=2)
    feedback_msg = gr.Textbox(label="系統訊息", interactive=False)

    # 改成如郭按下按鈕，就會重整聊天室，以免問題互相干擾
    submit_btn.click(
        answer_question,
        inputs=[user_input, chat_state],
        outputs=[chat_state, chat_display, current_answer, feedback_msg]
    )

    good_btn.click(
    record_feedback,
    inputs=[chat_state, gr.Textbox(value="是", visible=False), report_input],
    outputs=[chat_state, chat_display, feedback_msg]
    )

    bad_btn.click(
        record_feedback,
        inputs=[chat_state, gr.Textbox(value="否", visible=False), report_input],
        outputs=[chat_state, chat_display, feedback_msg]
    )


demo.launch()
