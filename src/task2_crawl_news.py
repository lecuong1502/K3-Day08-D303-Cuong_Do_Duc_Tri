"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Nguồn: trang công khai RMIT Vietnam (rmit.edu.vn) — News, Library events, Student events.

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Chạy:
    python task2_crawl_news.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


# 10 bài viết/thông báo thật từ RMIT Vietnam (sự kiện, thư viện, hỗ trợ sinh viên, tin tức)
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/libraryvn/about-us/library-events/2026/rmit-library-seminar-2026",
    "https://www.rmit.edu.vn/libraryvn/about-us/library-events/2026/r-loop-from-waste-to-value",
    "https://www.rmit.edu.vn/libraryvn/about-us/library-events/2026/beyond-the-pages",
    "https://www.rmit.edu.vn/libraryvn",
    "https://www.rmit.edu.vn/students/student-news-and-events",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-events-2026",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-events-2026/"
    "welcome-new-foundation-studies-students-sem-1-2026",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-events-2026/"
    "orientation-semester-1-2026",
    "https://www.rmit.edu.vn/news/english",
    "https://www.rmit.edu.vn/news/community",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title") or result.metadata.get("og:title") or "Unknown"

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
            "success": result.success,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    ok = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
            continue

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")

        status = "✓" if article.get("success", True) else "⚠ (crawl trả về nhưng success=False)"
        print(f"  {status} Saved: {filepath} — title: {article['title']}")
        ok += 1

    print(f"\nHoàn tất: {ok}/{len(ARTICLE_URLS)} bài crawl thành công.")


if __name__ == "__main__":
    asyncio.run(crawl_all())