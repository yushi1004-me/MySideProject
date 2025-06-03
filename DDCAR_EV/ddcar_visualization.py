import pandas as pd
import matplotlib.pyplot as plt

# 讀取已包含品牌提及次數的 CSV
df = pd.read_csv("ddcar_ev_news_with_brand.csv")

# 排除非品牌欄位
non_brand_cols = ["id", "新聞標題", "新聞內文", "提及的電動車品牌"]
brand_cols = [col for col in df.columns if col not in non_brand_cols and df[col].dtype in ["int64", "float64"]]

# 統計每個品牌總提及次數
brand_counts = df[brand_cols].sum().sort_values(ascending=False)

# === 1. 垂直長條圖（Bar Chart）===
plt.figure(figsize=(10, 6))
brand_counts.plot(kind='bar', color='cornflowerblue')
plt.title("Number of brands mentioned in the news", fontsize=14)
plt.ylabel("Number of Mentions", fontsize=12)
plt.xlabel("Brands", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("brand_mentions_bar.png")
plt.show()

# === 2. 圓餅圖（Pie Chart）===
plt.figure(figsize=(8, 8))
plt.pie(brand_counts, labels=brand_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Brand mention ratio", fontsize=14)
plt.axis('equal')  # 確保圓形
plt.tight_layout()
plt.savefig("brand_mentions_pie.png")
plt.show()
