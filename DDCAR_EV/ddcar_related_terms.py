# ddcar_related_terms.py

import pandas as pd
import time
import requests
import re 
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API 設定
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


# LLM prompt：找出品牌與車款
def build_prompt(content):
    return f"""
你是一個電動車分析助手。根據以下文章，請判斷是否提到了具體車款名稱，並將它們對應到所屬品牌（如 Tesla 的 Model Y）。
請用以下格式輸出：
品牌名: [車款1, 車款2, ...]

文章內容：
---
{content}
---
請注意：如果沒有提到任何具體車款，可以回傳空字典，例如：{{}}。
"""

# 呼叫 Groq API 分析
def call_groq(content, max_retries=3):
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "你是一位專業的電動車評論分析員。"},
            {"role": "user", "content": build_prompt(content)}
        ],
        "temperature": 0.3
    }

    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            if res.status_code == 429:
                wait_time = 5 * attempt
                print(f"⏳ 第 {attempt} 次：429 限流，等待 {wait_time} 秒再試...")
                time.sleep(wait_time)
                continue
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Groq API error (嘗試 {attempt}/{max_retries}):", e)
            time.sleep(5 * attempt)
    return ""


# 擷取所有車款名稱（來自每個 [] 區段）
def extract_models(llm_response):
    pattern = r"\[([^\]]+)\]"
    matches = re.findall(pattern, llm_response)
    models = []
    for match in matches:
        items = [x.strip() for x in match.split(",") if x.strip()]
        models.extend(items)
    # 去重後排序
    models = sorted(set(models))
    return "、".join(models) if models else ""

# 讀檔案
df = pd.read_csv("ddcar_ev_news_with_brand.csv")

llm_responses = []
car_model_tags = []

# 開始分析每一篇
for idx, row in df.iterrows():
    print(f"🔍 分析第 {idx+1} 篇文章...")
    content = str(row["新聞內文"])[:8000]  # 若內文太長可截斷
    llm_output = call_groq(content)
    llm_responses.append(llm_output)
    car_model_tags.append(extract_models(llm_output))
    time.sleep(5)

# 加欄位
df["品牌相關標籤（LLM）"] = llm_responses
df["相關車款列表"] = car_model_tags

# 輸出結果
df.to_csv("ddcar_ev_news_with_tags.csv", index=False, encoding="utf-8-sig")
print("✅ 已儲存結果至 ddcar_ev_news_with_tags.csv")
