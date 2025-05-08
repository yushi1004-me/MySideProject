import pandas as pd
import numpy as np
import faiss
import gradio as gr
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
import datetime

# ---------- 載入資源 ----------
df = pd.read_csv("faq_data.csv")
with open("faq_texts.pkl", "rb") as f:
    faq_texts = pickle.load(f)

index = faiss.read_index("faq.index")
model = SentenceTransformer("BAAI/bge-large-zh")
# 初始化 Groq API
client = Groq(api_key="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")


# ---------- 嵌入與回答 ----------
def embed(texts):
    return model.encode(
        [f"為這句話生成表示以用於檢索: {t}" for t in texts],
        normalize_embeddings=True
    )

def rephrase_answer(question, original_answer, chat_history):
    context = "。".join([q for q, _ in chat_history[-2:]])  # 取最近幾輪問題作為語境
    system_prompt = (
        "你是交通部的 AI 語言助理，請使用繁體中文，用自然、親切、專業的語氣回答民眾的問題。"
        "請根據下方提供的原始回答進行重寫，使其更易懂、更自然。"
        "不要出現『使用者：』或『AI：』等標籤，也不需要模仿對話格式。"
        "如果你無法回答問題，請使用這句話作為回應："
        "'對不起，您問的問題我目前無法回答，詳情請洽交通部客服專線詢問：0800-231-161。'"
    )

    user_prompt = (
        f"問題背景（近幾輪提問摘要）: {context}\n\n"
        f"這是原始回答：{original_answer}\n\n"
        f"請用繁體中文、自然語氣重新說明這段回答。"
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

# ---------- 查詢主流程 ----------
def answer_question(user_input, chat_history):
    query_vector = embed([user_input])[0].astype("float32").reshape(1, -1)
    D, I = index.search(query_vector, k=1)
    original_answer = df.iloc[I[0][0]]['答覆']
    rewritten = rephrase_answer(user_input, original_answer, chat_history)
    chat_history.append((user_input, rewritten))
    return chat_history, chat_history, rewritten, ""

# ---------- 回饋按鈕 ----------
def record_feedback(chat_history, helpful, report):
    timestamp = datetime.datetime.now().isoformat()
    last_q, last_a = chat_history[-1]
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\nQ: {last_q}\nA: {last_a}\n👍Helpful: {helpful}\n🐞Report: {report}\n\n")
    return "✅ 感謝您的回饋，已記錄！"

# ---------- Gradio 介面 ----------
with gr.Blocks() as demo:
    gr.Markdown("### 交通部 AI 問答助理（多輪對話 + 回饋）")

    chat_state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(label="請輸入您的問題", lines=2)
        submit_btn = gr.Button("送出問題")

    chat_display = gr.Chatbot(label="對話紀錄")
    current_answer = gr.Textbox(label="目前回覆", interactive=False)

    with gr.Row():
        good_btn = gr.Button("👍 滿意")
        bad_btn = gr.Button("👎 不滿意")

    error_report = gr.Textbox(label="如需補充說明錯誤，可在此填寫", lines=2)
    feedback_msg = gr.Textbox(label="系統訊息", interactive=False)

    submit_btn.click(
        answer_question,
        inputs=[user_input, chat_state],
        outputs=[chat_state, chat_display, current_answer, feedback_msg]
    )

    good_btn.click(
        record_feedback,
        inputs=[chat_state, gr.Textbox(value="是", visible=False), error_report],
        outputs=[feedback_msg]
    )

    bad_btn.click(
        record_feedback,
        inputs=[chat_state, gr.Textbox(value="否", visible=False), error_report],
        outputs=[feedback_msg]
    )

demo.launch()