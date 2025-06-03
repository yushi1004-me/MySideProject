# 用來「滑動到底」以觸發載入更多新聞
from selenium import webdriver
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
import time
import re
import datetime

# 1. 設定 Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 不開啟視窗
driver = webdriver.Chrome(options=options)

# 2. 打開即時新聞頁面
url = "https://www.ddcar.com.tw/news/categories/0/%E5%8D%B3%E6%99%82%E6%96%B0%E8%81%9E/list/"
driver.get(url)

# 3. 向下滑動 50 次，每次滑動後暫停 10 秒，讓網站有時間載入更多新聞（無限捲動機制）
last_height = driver.execute_script("return document.body.scrollHeight")
SCROLL_LIMIT = 50
for i in range(SCROLL_LIMIT):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(10)  # 每次滑動後等待 10 秒載入 JS 內容
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print(f"已滑到底，總共滑動 {i+1} 次")
        break
    last_height = new_height


# 4. 抓新聞連結
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
base_url = "https://www.ddcar.com.tw"
links = [a["href"] for a in soup.select("a.title.my-2") if a.has_attr("href")]
print(f"共擷取文章數：{len(links)}")


# 5. 設定品牌關鍵字
brands = ["Tesla", "Hyundai", "BMW i", "Lexus", "Nissan", "Toyota", "Mercedes", "Porsche"]
brand_pattern = re.compile("|".join(brands), re.IGNORECASE)

# 6. 抓取每篇新聞並統計品牌名稱出現次數
data = []
# 使用今天日期 + 編號當作主鍵 id
today = datetime.date.today().strftime("%Y%m%d")

for idx, link in enumerate(links, 1):
    try:
        driver.set_page_load_timeout(30)  # 設定單篇最大等待 30 秒
        driver.get(link)
        time.sleep(4)

        article_html = driver.page_source
        article_soup = BeautifulSoup(article_html, "html.parser")

        # 標題與內文
        title_tag = article_soup.select_one("h1") or article_soup.select_one("title")
        title = title_tag.get_text(strip=True) if title_tag else "無標題"
        
        # 改成只抓乾淨的 <p> 段落文字，忽略含有圖片的，且濾掉含圖或廣告段的內文
        paragraphs = article_soup.find_all("p")
        clean_paragraphs = []
        stop_phrases = ["推薦閱讀", "DDCAR 有 LINE", "bit.ly", "ZeroWidthSpace"]

        for p in paragraphs:
            # 忽略含 <img> 的段落
            if p.find("img"):
                continue

            text = p.get_text(strip=True)

            # 若碰到廣告相關文字，則提早停止收集（避免抓到底部推薦段）
            if any(phrase in text for phrase in stop_phrases):
                break

            if text:
                clean_paragraphs.append(text)

        # 將有效段落組成一段內文
        content = "\n".join(clean_paragraphs)

        # 篩選掉錯誤頁面（常見 400 頁面內容或錯誤訊息）
        if "HTTP 400" in content or "找不到網頁" in content or "無法顯示頁面" in content or len(content.strip()) < 100:
            print(f"⚠️ 無效文章跳過：{link}")
            continue

        # 統計品牌
        matches = brand_pattern.findall(content)
        match_counter = Counter([m.title() if m.lower() != "bmw i" else "BMW i" for m in matches])
        mentioned_brands = "、".join(sorted(set(match_counter.keys()), key=lambda x: brands.index(x))) if match_counter else ""

        # 構建資料行
        row = {
            "id": f"{today}{idx:02d}",
            "新聞標題": title,
            "新聞內文": content.replace("\\n", "").replace("\\r", "").strip(),
            "提及的電動車品牌": mentioned_brands,
            "連結": link
        }
        for brand in brands:
            row[brand] = match_counter.get(brand, 0)

        data.append(row)

    except Exception as e:
        print(f"抓取失敗 {link}：{e}")
        continue

driver.quit()

# 7. 統計結果輸出為 CSV
df = pd.DataFrame(data)
df.to_csv("ddcar_ev_news.csv", index=False, encoding="utf-8-sig")
print("已儲存為 ddcar_ev_news.csv")

