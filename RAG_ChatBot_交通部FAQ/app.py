import pandas as pd
import numpy as np
import faiss
import gradio as gr
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------- Step 1: 載入資料 ----------

df = pd.read_csv("faq_data.csv")
with open("faq_texts.pkl", "rb") as f:
    faq_texts = pickle.load(f)

index = faiss.read_index("faq.index")
model = SentenceTransformer("BAAI/bge-large-zh")

# 初始化 Groq API
client = Groq(api_key="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")


# ---------- Step 2: 查詢與改寫函式 ----------

def embed(texts):
    return model.encode(
        [f"為這句話生成表示以用於檢索: {t}" for t in texts],
        normalize_embeddings=True
    )

def rephrase_answer(question, original_answer):
    system_prompt = (
        "你是交通部的 AI 語言助理，負責以繁體中文、口語、友善且專業的方式回答民眾的問題。"
        "請根據下方提供的原始回答，轉述為更自然易懂的說法，語氣應保持專業與親切。"
        "所有回覆都必須使用繁體中文。"
        "如果你無法回答問題或沒有足夠資訊，請直接回答："
        "'對不起，您問的問題我目前無法回答，詳情請洽交通部客服專線詢問：0800-231-161。'"
    )

    user_prompt = (
        f"使用者問題：{question}\n\n"
        f"原始回覆：{original_answer}\n\n"
        f"請重新用繁體中文轉述，使其更自然易懂。"
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


def answer_question(user_input):
    query_vector = embed([user_input])[0].astype("float32").reshape(1, -1)
    D, I = index.search(query_vector, k=1)
    original_answer = df.iloc[I[0][0]]['答覆']
    rewritten_answer = rephrase_answer(user_input, original_answer)
    return rewritten_answer


# ---------- Step 3: Gradio UI ----------

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, label="請輸入您的問題"),
    outputs=gr.Textbox(lines=5, label="交通部 AI 回覆"),
    title="交通部 AI 問答助理",
    description="輸入您想詢問的內容，我們將以自然、友善又專業的方式為您解答。"
)

demo.launch()
