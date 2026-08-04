"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Nguồn: trang công khai RMIT Vietnam (rmit.edu.vn).
Cả 3 link dưới đây là link PDF trực tiếp (không cần đăng nhập, không bị 403),
đã kiểm tra tồn tại tại thời điểm viết script.

Chạy: python task1_collect_legal_docs.py
"""

from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# (filename, url, mo_ta)
DOCUMENTS = [
    (
        "tuition-fees-rmit.pdf",
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/"
        "documents/pdfs/study-at-rmit/tuition-fees/student-fees-and-charges-guide-06-2026.pdf",
        "Student Fees & Charges Guide 2026 - hoc phi & phuong thuc thanh toan",
    ),
    (
        "academic-achievement-scholarship-rmit.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/study-at-rmit/scholarships/english-pdf/"
        "rmit-university-vietnam-scholarship-terms-and-conditions.pdf",
        "RMIT Vietnam Scholarship Terms and Conditions - chinh sach hoc bong",
    ),
    (
        "accommodation-services-rmit.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/students/accommodation/"
        "accommodation-advice-for-international-students-in-vietnam.pdf",
        "Accommodation Advice for International Students - ky tuc xa / cho o",
    ),
    (
        "course-registration-guide-rmit.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/students/myrmit-qrg/myrmit-enrolment-how-to-enrol-qrg.pdf",
        "myRMIT Enrolment - How to Enrol QRG - huong dan dang ky hoc phan",
    ),
    (
        "enrolment-procedure-policy-rmit.pdf",
        "https://policies.rmit.edu.au/download.php?id=113&version=3",
        "Enrolment Procedure (RMIT Policies) - quy dinh dang ky nhap hoc",
    ),
    (
        "national-medical-insurance-guide-rmit.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/students/advice-support/"
        "national-medical-insurance-guide-for-new-students.pdf",
        "National Medical Insurance Guide for New Students - bao hiem y te",
    ),
    (
        "undergraduate-programs-brochure-rmit.pdf",
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/"
        "pdfs/study-at-rmit/programs/english-pdf/undergraduate-programs/"
        "undergraduate-brochure-en-2026.pdf",
        "Undergraduate Programs Brochure 2026 - cam nang chuong trinh dai hoc",
    ),
    (
        "international-student-guide-rmit.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/study-at-rmit/international-students/"
        "international-student-guide-2024.pdf",
        "International Student Guide 2024-2025 - cam nang sinh vien quoc te",
    ),
    (
        "predeparture-guide-rmit.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/student-life/international-student-support/insurance/"
        "predeparture-guide-for-study-abroad-students.pdf",
        "Pre-departure Guide for Study Abroad Students - huong dan truoc nhap hoc",
    ),
    (
        "rmit-vietnam-factsheet-rmit.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/global-experiences/rmit-vietnam-factsheet-2024-2025.pdf",
        "RMIT Vietnam Fact Sheet 2024-2025 - tong quan enrolment & ho tro sv",
    ),
]

HEADERS = {
    # Một số CDN của RMIT chặn request không có User-Agent trình duyệt
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str, description: str) -> bool:
    filepath = DATA_DIR / filename
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("Content-Type", "").lower() and not resp.content.startswith(b"%PDF"):
            print(f"⚠ Cảnh báo: {filename} có thể không phải PDF hợp lệ (Content-Type: {resp.headers.get('Content-Type')})")
        filepath.write_bytes(resp.content)
        size_kb = filepath.stat().st_size / 1024
        print(f"✓ Đã tải: {filepath} ({size_kb:.1f} KB) — {description}")
        return True
    except requests.RequestException as e:
        print(f"✗ Lỗi khi tải {filename}: {e}")
        print(f"  → Nếu bị chặn (403/timeout), tải thủ công từ: {url}")
        return False


def main() -> None:
    setup_directory()
    ok = 0
    for filename, url, description in DOCUMENTS:
        if download_file(url, filename, description):
            ok += 1
    print(f"\nHoàn tất: {ok}/{len(DOCUMENTS)} file tải thành công.")


if __name__ == "__main__":
    main()