"""
Pytest conftest — thêm cả project root và src/ vào sys.path.

Lý do cần cả 2 đường dẫn:
- Test dùng import kiểu package: `from src.task5_semantic_search import ...`
  → cần PROJECT ROOT trong sys.path (để "src" resolve được như 1 package,
  với điều kiện có src/__init__.py).
- Bên trong các file task*.py lại tự import lẫn nhau kiểu FLAT (không có
  prefix "src."), vd task5_semantic_search.py có dòng
  `from task4_chunking_indexing import ...` → cần chính SRC/ nằm trong
  sys.path để dòng import nội bộ đó resolve được.

Thiếu 1 trong 2 đường dẫn sẽ gây ImportError khi pytest cố import module,
khiến các test liên quan bị SKIP (theo cơ chế try/except trong test).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"

for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)