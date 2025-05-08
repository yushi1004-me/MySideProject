# 要先安裝transformers 和 sentence-transformers
# pip install transformers sentence-transformers

import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 讀取資料
df = pd.read_csv("/Users/hsuhuiyu/Documents/碩一下/資訊系統專案管理/data_finalproject/交通部常見問答集_清理版.csv")

# 載入模型
model = SentenceTransformer('BAAI/bge-large-zh')

# 合併問題與答案
faq_texts = [f"Q: {row['問題']} A: {row['答覆']}" for _, row in df.iterrows()]

# 向量嵌入函式
def embed(texts):
    return model.encode(
        [f"為這句話生成表示以用於檢索: {t}" for t in texts],
        normalize_embeddings=True
    )

# 產生向量嵌入
faq_embeddings = embed(faq_texts)
embedding_matrix = np.array(faq_embeddings).astype("float32")

# 建立向量索引
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# 儲存 FAISS index
faiss.write_index(index, "faq.index")

# 儲存 FAQ 文字內容
with open("faq_texts.pkl", "wb") as f:
    pickle.dump(faq_texts, f)

# 儲存原始資料
df.to_csv("faq_data.csv", index=False)

print("已完成向量建立並儲存 faq.index、faq_texts.pkl 和 faq_data.csv")
