import pandas as pd
from collections import Counter
import re

# Step 1: 載入內文資料
df = pd.read_csv("ddcar_ev_news.csv")

# Step 2: 定義品牌對照表
brand_map = {
    "Tesla": ["Tesla", "特斯拉"],
    "Nissan": ["Nissan", "日產"],
    "Hyundai": ["Hyundai", "現代"],
    "BYD": ["BYD", "比亞迪"],
    "BMW": ["BMW", "BMW i"],
    "Lexus": ["Lexus", "雷克薩斯"],
    "Mercedes": ["Mercedes", "賓士", "Mercedes-Benz"],
    "Volkswagen": ["Volkswagen", "福斯", "VW"],
    "Audi": ["Audi", "奧迪"],
    "Toyota": ["Toyota", "豐田"],
    "Kia": ["Kia", "起亞"],
    "Porsche": ["Porsche", "保時捷"],
    "Mazda": ["Mazda", "馬自達"],
    "Ford": ["Ford", "福特"],
    "MG": ["MG", "名爵"],
    "Volvo": ["Volvo", "富豪"],
    "Peugeot": ["Peugeot", "標緻"],
    "Renault": ["Renault", "雷諾"],
    "Lucid": ["Lucid"],
    "Rivian": ["Rivian"],
    "Honda": ["Honda", "本田"]
}

# Step 3: 初始化欄位
brand_names = list(brand_map.keys())
df["提及的電動車品牌"] = ""

for brand in brand_names:
    df[brand] = 0  # 初始化每個品牌欄位

# Step 4: 分析每一篇文章
for idx, row in df.iterrows():
    content = str(row["新聞內文"]).lower()
    mention_counts = Counter()
    mentioned_brands = []

    for brand, aliases in brand_map.items():
        count = sum(content.count(alias.lower()) for alias in aliases)
        if count > 0:
            mention_counts[brand] = count
            mentioned_brands.append(brand)
            df.at[idx, brand] = count

    df.at[idx, "提及的電動車品牌"] = "、".join(mentioned_brands)

# Step 5: 輸出結果
df.to_csv("ddcar_ev_news_with_brand.csv", index=False, encoding="utf-8-sig")
print("已輸出至 ddcar_ev_news_with_brand.csv")
