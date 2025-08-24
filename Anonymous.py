import os
import re
from pathlib import Path

HIDDEN_DIRS = {'.git', '.svn', '.hg', '.idea', '.vscode', '__pycache__'}

def _is_probably_binary(path: Path, chunk_size: int = 4096) -> bool:
    with path.open('rb') as f:
        chunk = f.read(chunk_size)
    if not chunk:
        return False
    if b'\x00' in chunk:
        return True
    text_like = sum(b in b"\n\r\t\f\b\x07\x08\x0c\x1b" or 32 <= b <= 126 for b in chunk)
    return (text_like / len(chunk)) < 0.90

def _iter_text_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in HIDDEN_DIRS and not d.startswith('.')]
        for name in filenames:
            p = Path(dirpath) / name
            if not p.is_file() or p.is_symlink():
                continue
            if _is_probably_binary(p):
                continue
            yield p

def scan_occurrences(old: str, root: str = "."):
    """
    不区分大小写扫描 root 及子目录所有文本文件中 old 的出现情况。
    返回：[(path, count), ...]（仅包含命中文件），并打印汇总。
    """
    if not old:
        print("⚠️ 旧串为空，未执行扫描。")
        return []

    pattern = re.compile(re.escape(old), flags=re.IGNORECASE)
    hits = []
    total = 0
    for p in _iter_text_files(root):
        text = p.read_text(encoding="utf-8", errors="ignore")
        cnt = len(pattern.findall(text))
        if cnt:
            hits.append((str(p), cnt))
            total += cnt
    for path, cnt in hits:
        print(f"[HIT] {path}  ×{cnt}")
    print(f"—— 扫描完成：命中文件 {len(hits)} 个，总命中 {total} 处 ——")
    return hits

def replace_in_tree(old: str, new: str, root: str = "."):
    """
    不区分大小写地在 root 及子目录所有“文本文件”中，把 old 替换为 new。
    """
    if not old:
        print("⚠️ 旧串为空，未执行。")
        return
    # old == new 也可能需要替换大小写，为避免无意义操作，这里仍然跳过：
    if old.lower() == new.lower():
        print("ℹ️ 旧/新字符串在不区分大小写时一致，可能无需处理（已跳过）。")
        return

    pattern = re.compile(re.escape(old), flags=re.IGNORECASE)

    changed_files = 0
    total_replacements = 0
    for p in _iter_text_files(root):
        text = p.read_text(encoding="utf-8", errors="ignore")
        new_text, cnt = pattern.subn(new, text)
        if cnt == 0:
            continue
        p.write_text(new_text, encoding="utf-8")
        changed_files += 1
        total_replacements += cnt
        print(f"[WRITE] {p}  替换 {cnt} 处")
    print(f"—— 替换完成：修改文件 {changed_files} 个，替换总次数 {total_replacements} ——")

# 示例：匿名化作者（大小写不敏感）
def anonymize_author(root: str = "."):
    target = "# Author : yuxiang Zeng"
    scan_occurrences(target, root)
    replace_in_tree(target, "# Author : Anonymous", root)

# 直接执行替换（示例）
replace_in_tree("# Author : yuxiang Zeng", "# Author : Anonymous", ".")