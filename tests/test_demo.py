from datetime import date

from jobagent import config
from jobagent.demo import DailyLimit, HashEmbeddings, build_index


def test_hash_embeddings_are_unit_vectors_and_similar_for_shared_words():
    e = HashEmbeddings(size=256)
    a, b, c = e.embed_query("大模型评测 Python"), e.embed_query("Python 评测流水线"), e.embed_query("cooking recipes")
    assert abs(sum(x * x for x in a) - 1) < 1e-9
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(a, b) > dot(a, c)


def test_daily_limit_resets_each_day(tmp_path):
    limit = DailyLimit(tmp_path / "usage.json", 2)
    assert limit.use() and limit.use() and not limit.use() and limit.remaining() == 0
    (tmp_path / "usage.json").write_text('{"day": "2000-01-01", "used": 99}', encoding="utf-8")
    assert limit.remaining() == 2 and date.today().isoformat() != "2000-01-01"


def test_demo_profile_is_fictional_and_searchable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_PROFILE_DIR", "demo/profile")
    index, name = build_index(tmp_path)
    hits = index.search("LoRA 微调", k=3)
    assert any("LoRA" in h.page_content for h in hits)
    text = " ".join(h.page_content for h in index.search("示例 虚构", k=5))
    assert "虚构" in text or "FICTIONAL" in text
