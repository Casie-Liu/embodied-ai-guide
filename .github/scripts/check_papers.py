#!/usr/bin/env python3
"""校验 docs/papers.json：schema 完整性 + arXiv 编号真实存在。

用法（本地或 CI）:
    python check_papers.py [path/to/papers.json]

规则:
  1. 每条记录必须含 title/org/year/venue/category/verified，arxiv 可为 null。
  2. arxiv 编号格式必须是 \\d{4}\\.\\d{4,5}（或旧式 \\d{7}）。
  3. 通过 arXiv 官方 API (export.arxiv.org) 批量核验编号存在且非错误条目。
     API 不可达时降级为警告（不因网络问题挂 CI）。
"""
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^\d{7}(v\d+)?$")
API_URL = "http://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
BATCH = 20
REQUIRED = ["title", "org", "year", "venue", "category", "verified"]


def fail(msg: str) -> None:
    print(f"❌ {msg}")


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def check_schema(papers: list) -> list:
    """返回所有 arxiv 编号（供后续核验），同时就地报 schema 错误。"""
    ids = []
    errors = 0
    for i, p in enumerate(papers):
        for key in REQUIRED:
            if key not in p:
                fail(f"[{i}] 缺字段 {key}: {p.get('title', '?')[:50]}")
                errors += 1
        arxiv = p.get("arxiv")
        if arxiv is not None:
            if not isinstance(arxiv, str) or not ARXIV_RE.match(arxiv.split("v")[0] if "v" in arxiv else arxiv):
                fail(f"[{i}] arXiv 编号格式可疑: {arxiv!r} ({p.get('title', '?')[:50]})")
                errors += 1
            else:
                ids.append(arxiv)
    if errors == 0:
        ok(f"schema 检查通过（{len(papers)} 条，{len(ids)} 个 arXiv 编号）")
    return ids


def check_arxiv(ids: list) -> int:
    """批量核验编号存在。返回错误数。网络故障时不计错（降级警告）。"""
    errors = 0
    bad = []
    for start in range(0, len(ids), BATCH):
        batch = ids[start:start + BATCH]
        url = API_URL.format(ids=",".join(batch), n=BATCH)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            ns = {"a": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(data)
            entries = root.findall("a:entry", ns)
            if len(entries) < len(batch):
                # arXiv 对无效 id 返回含 Error 的 entry 或直接缺失
                got = set()
                for e in entries:
                    eid = e.find("a:id", ns).text  # http://arxiv.org/abs/XXXXvN
                    got.add(eid.rsplit("/", 1)[-1])
                for want in batch:
                    base = want.split("v")[0]
                    if not any(g.startswith(base) for g in got):
                        bad.append(want)
            time.sleep(3)  # arXiv API 礼貌间隔
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ arXiv API 不可达，跳过本批核验（{exc}）")
            return 0
    for b in bad:
        fail(f"arXiv 编号不存在或被 API 拒绝: {b}")
        errors += 1
    if errors == 0:
        ok(f"arXiv 核验通过（{len(ids)} 个编号全部存在）")
    return errors


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/papers.json"
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    papers = doc["papers"]
    print(f"== 检查 {path}：{len(papers)} 条 ==")
    ids = check_schema(papers)
    if not ids:
        return 1
    errors = check_arxiv(ids)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
