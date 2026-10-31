"""Web tools without the internet: httpx.MockTransport stands in for websites and search APIs."""
import json

import httpx
import pytest

from jobagent import web
from jobagent.schemas import BriefPoint, CompanyBrief, SearchPlan
from jobagent.tools import build_tools
from tests.fakes import FakeLLM
from tests.test_agent import JD, env, handlers  # noqa: F401

PAGE = "<html><body><nav>menu</nav><main><h1>大模型评测工程师</h1><p>" + "岗位职责：设计评测集与评分标准。" * 30 + \
       "</p><p>公司：Huawei</p></main><script>var x=1;</script></body></html>"


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_fetch_page_extracts_text():
    client = mock_client(lambda req: httpx.Response(200, text=PAGE))
    text = web.fetch_page("https://jobs.example.com/1", client=client)
    assert "评分标准" in text and "var x" not in text


def test_fetch_page_errors(monkeypatch):
    monkeypatch.setattr(web, "_render_with_browser", lambda url: None)   # no Playwright
    js_app = mock_client(lambda req: httpx.Response(200, text="<html><div id=app></div></html>"))
    with pytest.raises(web.WebError, match="JavaScript"):
        web.fetch_page("https://career.example.com/job", client=js_app)
    gone = mock_client(lambda req: httpx.Response(404))
    with pytest.raises(web.WebError, match="Could not open"):
        web.fetch_page("https://example.com/x", client=gone)
    with pytest.raises(web.WebError, match="web link"):
        web.fetch_page("career.huawei.com")


def test_search_providers(monkeypatch):
    monkeypatch.setattr(web.config, "SEARCH_API_KEY", "k")
    seen = {}

    def bocha(req):
        seen["bocha"] = (str(req.url), json.loads(req.content), req.headers["authorization"])
        return httpx.Response(200, json={"data": {"webPages": {"value": [
            {"name": "华为诺亚方舟实验室", "url": "https://a.example", "summary": "大模型研究", "datePublished": "2026-09-01"}]}}})

    monkeypatch.setattr(web.config, "SEARCH_PROVIDER", "bocha")
    r = web.search("华为 大模型", client=mock_client(bocha))
    assert r == [{"title": "华为诺亚方舟实验室", "url": "https://a.example", "snippet": "大模型研究", "date": "2026-09-01"}]
    assert seen["bocha"][0].endswith("/v1/web-search") and seen["bocha"][2] == "Bearer k"

    monkeypatch.setattr(web.config, "SEARCH_PROVIDER", "tavily")
    tav = mock_client(lambda req: httpx.Response(200, json={"results": [{"title": "T", "url": "https://t", "content": "c"}]}))
    assert web.search("q", client=tav)[0]["snippet"] == "c"

    bad = mock_client(lambda req: httpx.Response(401))
    with pytest.raises(web.WebError, match="HTTP 401"):
        web.search("q", client=bad)
    monkeypatch.setattr(web.config, "SEARCH_API_KEY", "")
    with pytest.raises(web.WebError, match="not set up"):
        web.search("q")


def test_fetch_job_and_research_tools(env, monkeypatch):
    index, store = env
    h = handlers("Huawei")
    h[SearchPlan] = lambda p: SearchPlan(queries=["华为 大模型评测 团队", "Huawei LLM evaluation team"])
    h[CompanyBrief] = lambda p: CompanyBrief(
        overview=[BriefPoint(text="Builds the Pangu models", sources=[1]),
                  BriefPoint(text="Unsourced rumour", sources=[])],          # dropped: no source
        recent=[BriefPoint(text="New eval benchmark released", sources=[2, 9])],
        emphasize=["multilingual evaluation"], questions=["How are eval sets versioned?"])
    llm = FakeLLM([], h)
    tools = {t.name: t for t in build_tools(index, store, llm)}

    monkeypatch.setattr(web, "fetch_page", lambda url: JD * 20)
    out = tools["fetch_job"].invoke({"url": "https://career.example.com/job/1"})
    jid = json.loads(out.split("\n", 1)[1])["job_id"]
    assert store.get(jid)["url"] == "https://career.example.com/job/1"

    calls = []
    monkeypatch.setattr(web, "search", lambda q, count=5: calls.append(q) or [
        {"title": f"R{len(calls)}", "url": f"https://s/{len(calls)}", "snippet": "x", "date": ""},
        {"title": "dup", "url": "https://s/1", "snippet": "x", "date": ""}])
    out = tools["research_company"].invoke({"job_id": jid})
    assert len(calls) == 2
    assert "Builds the Pangu models [1]" in out and "Unsourced rumour" not in out
    assert "[2]" in out and "[9]" not in out                              # invented source number removed
    assert "## Sources" in out and store.get(jid)["research"]["emphasize"] == ["multilingual evaluation"]

    def broken(url):
        raise web.WebError("The page loads its content with JavaScript")
    monkeypatch.setattr(web, "fetch_page", broken)
    assert "Could not read the posting" in tools["fetch_job"].invoke({"url": "https://x"})


def test_find_jobs_filters_and_ranks(env, monkeypatch):
    from jobagent.schemas import JobAnalysis, JobCandidate, JobCandidates, Requirement

    index, store = env
    results = [
        {"title": "大模型评测入门 - 博客园", "url": "https://blog/1", "snippet": "教程", "date": ""},
        {"title": "华为 大模型评测工程师 校招", "url": "https://jobs/huawei", "snippet": "岗位职责", "date": ""},
        {"title": "字节 评测研发 (已截止)", "url": "https://jobs/closed", "snippet": "已截止", "date": ""},
        {"title": "腾讯 大模型评测 2027届", "url": "https://jobs/tencent", "snippet": "岗位要求", "date": ""},
        {"title": "某公司 评测 2027届", "url": "https://jobs/old", "snippet": "岗位要求", "date": ""},
        {"title": "Saved before", "url": "https://jobs/known", "snippet": "", "date": ""},
    ]
    store.put({"id": "known123", "url": "https://jobs/known", "analysis": {}})
    monkeypatch.setattr(web, "search", lambda q, count=8, **kw: results)
    pages = {"https://jobs/huawei": "HUAWEI " + JD * 20, "https://jobs/tencent": "TENCENT " + JD * 20,
             "https://jobs/old": "OLD " + JD * 20}
    fetched = []
    monkeypatch.setattr(web, "fetch_page", lambda url: fetched.append(url) or pages[url])

    def analysis(prompt):
        company = "Huawei" if "HUAWEI" in prompt else "Tencent" if "TENCENT" in prompt else "Old Co"
        reqs = [Requirement(text="Python", type="must")]
        if company == "Tencent":
            reqs.append(Requirement(text="Kubernetes", type="must"))   # lower score
        return JobAnalysis(company=company, title=f"{company} LLM Eval", location="Shanghai", language="Chinese",
                           deadline="2020-01-01" if company == "Old Co" else "", requirements=reqs)

    h = handlers()
    h[JobAnalysis] = analysis
    h[SearchPlan] = lambda p: SearchPlan(queries=["2027届 大模型评测 上海", "LLM evaluation Shanghai 2027"])
    h[JobCandidates] = lambda p: JobCandidates(results=[
        JobCandidate(index=1, is_posting=False, is_open=True, relevant=True, reason="tutorial blog"),
        JobCandidate(index=2, is_posting=True, is_open=True, relevant=True),
        JobCandidate(index=3, is_posting=True, is_open=True, relevant=True),   # snippet says 已截止
        JobCandidate(index=4, is_posting=True, is_open=True, relevant=True),
        JobCandidate(index=5, is_posting=True, is_open=True, relevant=True),   # deadline passed
        JobCandidate(index=6, is_posting=True, is_open=True, relevant=True),   # already known
        JobCandidate(index=42, is_posting=True, is_open=True, relevant=True),  # no such result
    ])
    llm = FakeLLM([], h)
    tools = {t.name: t for t in build_tools(index, store, llm)}

    out = tools["find_jobs"].invoke({"request": "大模型评测 上海 2027届", "limit": 5})
    assert fetched == ["https://jobs/huawei", "https://jobs/tencent", "https://jobs/old"]
    assert out.index("Huawei") < out.index("Tencent")                     # ranked by match score
    assert "Old Co" not in out.split("Notes:")[0]
    assert "blog" not in out and "jobs/closed" not in out and "jobs/known" not in out.split("Notes:")[0]
    assert "Nothing was added to the tracker" in out and store.tracker() == []
    report = next(store.reports_dir.glob("search-*-jobs.md")).read_text(encoding="utf-8")
    assert "## Skipped results" in report
    assert "not a single posting" in report and "closed" in report and "already analyzed" in report
    assert "deadline passed" in report

    fetched.clear()
    out = tools["find_jobs"].invoke({"request": "x", "limit": 1})        # limit respected; known jobs skipped now
    assert len(fetched) <= 1


def test_find_jobs_without_search_key(env, monkeypatch):
    index, store = env
    monkeypatch.setattr(web.config, "SEARCH_API_KEY", "")
    h = handlers()
    h[SearchPlan] = lambda p: SearchPlan(queries=["q"])
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], h))}
    assert "not set up" in tools["find_jobs"].invoke({"request": "LLM jobs"})


def test_find_jobs_city_is_preference_and_rejections_are_explained(env, monkeypatch):
    from jobagent.schemas import JobAnalysis, JobCandidate, JobCandidates, Requirement

    index, store = env
    results = [{"title": "评测工程师 北京", "url": "https://jobs/bj", "snippet": "", "date": ""},
               {"title": "评测工程师 上海", "url": "https://jobs/sh", "snippet": "", "date": ""},
               {"title": "前端开发", "url": "https://jobs/fe", "snippet": "", "date": ""}]
    monkeypatch.setattr(web, "search", lambda q, count=8, **kw: results)
    monkeypatch.setattr(web, "fetch_page", lambda url: url + " " + JD * 20)
    fetched = []
    real = web.fetch_page
    monkeypatch.setattr(web, "fetch_page", lambda url: fetched.append(url) or real(url))
    h = handlers("Co")
    h[SearchPlan] = lambda p: SearchPlan(queries=["q"])
    h[JobCandidates] = lambda p: JobCandidates(results=[
        JobCandidate(index=1, is_posting=True, is_open=True, relevant=True, fits_city_year=False),
        JobCandidate(index=2, is_posting=True, is_open=True, relevant=True, fits_city_year=True),
        JobCandidate(index=3, is_posting=True, is_open=True, relevant=False, reason="frontend role")])
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], h))}
    out = tools["find_jobs"].invoke({"request": "评测 上海", "limit": 1})
    assert fetched == ["https://jobs/sh"]                    # city match read first, Beijing still allowed
    report = next(store.reports_dir.glob("search-*-jobs.md")).read_text(encoding="utf-8")
    assert "role does not fit: frontend role" in report


def test_verification_pages_are_not_jobs(monkeypatch):
    monkeypatch.setattr(web, "_render_with_browser", lambda url: None)
    page = "<html><body><h1>访问验证</h1><p>" + "请拖动滑块完成验证。" * 60 + "</p></body></html>"
    client = mock_client(lambda req: httpx.Response(200, text=page))
    with pytest.raises(web.WebError, match="verification"):
        web.fetch_page("https://www.liepin.com/job/1", client=client)


def test_company_from_search_result(env, monkeypatch):
    from jobagent.schemas import JobAnalysis, JobCandidate, JobCandidates, Requirement

    index, store = env
    monkeypatch.setattr(web, "search", lambda q, count=8, **kw: [
        {"title": "大模型评测 - 牛客", "url": "https://jobs/a", "snippet": "", "date": ""}])
    monkeypatch.setattr(web, "fetch_page", lambda url: JD * 20)
    h = handlers("")                                          # the page text names no company
    h[JobAnalysis] = lambda p: JobAnalysis(company="", title="评测工程师", location="上海", language="Chinese",
                                           requirements=[Requirement(text="Python", type="must")])
    h[SearchPlan] = lambda p: SearchPlan(queries=["q"])
    h[JobCandidates] = lambda p: JobCandidates(results=[
        JobCandidate(index=1, is_posting=True, is_open=True, relevant=True, company="小红书")])
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], h))}
    out = tools["find_jobs"].invoke({"request": "评测"})
    assert "| 小红书 |" in out


def test_baidu_search(monkeypatch):
    monkeypatch.setattr(web.config, "SEARCH_API_KEY", "bce-v3/k")
    monkeypatch.setattr(web.config, "SEARCH_PROVIDER", "baidu")
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if req.url.path.endswith("/web_search"):
            return httpx.Response(404)                 # falls back to the older path
        body = json.loads(req.content)
        assert body["messages"][0]["content"] == "上海 大模型评测 校招" and req.headers["authorization"] == "Bearer bce-v3/k"
        return httpx.Response(200, json={"request_id": "x", "references": [
            {"id": 1, "title": "大模型评测工程师-牛客", "url": "https://n/1", "content": "岗位职责", "date": "2026-10-01"}]})

    r = web.search("上海 大模型评测 校招", client=mock_client(handler))
    assert r == [{"title": "大模型评测工程师-牛客", "url": "https://n/1", "snippet": "岗位职责", "date": "2026-10-01"}]
    assert calls[0].endswith("/v2/ai_search/web_search") and calls[1].endswith("/v2/ai_search")

    err = mock_client(lambda req: httpx.Response(200, json={"code": 216003, "message": "quota exceeded"}))
    with pytest.raises(web.WebError, match="quota exceeded"):
        web.search("q", client=err)


def test_search_on_job_sites(monkeypatch):
    monkeypatch.setattr(web.config, "SEARCH_API_KEY", "k")
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"references": [], "results": [], "data": {}})

    for provider in ("baidu", "tavily", "bocha"):
        monkeypatch.setattr(web.config, "SEARCH_PROVIDER", provider)
        web.search("大模型评测 校招", client=mock_client(handler), sites=["www.nowcoder.com", "www.zhipin.com"],
                   recency="semiyear")
    baidu, tavily, bocha = bodies
    assert baidu["search_filter"] == {"match": {"site": ["www.nowcoder.com", "www.zhipin.com"]}}
    assert baidu["search_recency_filter"] == "semiyear"
    assert tavily["include_domains"] == ["www.nowcoder.com", "www.zhipin.com"] and tavily["days"] == 183
    assert bocha["include"] == "www.nowcoder.com|www.zhipin.com"


def test_find_jobs_uses_job_sites_then_the_web(env, monkeypatch):
    index, store = env
    calls = []
    monkeypatch.setattr(web, "search", lambda q, count=8, sites=None, recency="year": calls.append(sites) or [])
    monkeypatch.setattr(web.config, "JOB_SITES", ["www.nowcoder.com"])
    h = handlers()
    h[SearchPlan] = lambda p: SearchPlan(queries=["a", "b"])
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], h))}
    tools["find_jobs"].invoke({"request": "评测"})
    assert calls == [["www.nowcoder.com"], ["www.nowcoder.com"], None]   # few results -> one open-web search


def test_find_jobs_skips_pages_without_a_job(env, monkeypatch):
    from jobagent.schemas import JobAnalysis, JobCandidate, JobCandidates

    index, store = env
    monkeypatch.setattr(web, "search", lambda q, count=8, **kw: [
        {"title": "猎聘", "url": "https://jobs/empty", "snippet": "", "date": ""}])
    monkeypatch.setattr(web, "fetch_page", lambda url: JD * 20)
    h = handlers("Co")
    h[JobAnalysis] = lambda p: JobAnalysis(company="", title="", location="", language="Chinese", requirements=[])
    h[SearchPlan] = lambda p: SearchPlan(queries=["q"])
    h[JobCandidates] = lambda p: JobCandidates(results=[
        JobCandidate(index=1, is_posting=True, is_open=True, relevant=True)])
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], h))}
    out = tools["find_jobs"].invoke({"request": "评测"})
    assert "jobs/empty" not in out.split("Notes:")[0]
    report = next(store.reports_dir.glob("search-*-jobs.md")).read_text(encoding="utf-8")
    assert "no job description on the page" in report


def test_check_page_is_retried_in_the_browser(monkeypatch):
    check = "<html><body><h1>安全验证</h1><p>" + "请完成验证。" * 80 + "</p></body></html>"
    job = "<html><body><h1>大模型评测工程师</h1><p>" + "岗位职责：负责大模型评测体系建设。" * 30 + "</p></body></html>"
    monkeypatch.setattr(web, "_render_with_browser", lambda url: job)       # logged-in browser sees the job
    text = web.fetch_page("https://www.zhipin.com/job_detail/1.html",
                          client=mock_client(lambda req: httpx.Response(200, text=check)))
    assert "岗位职责" in text

    monkeypatch.setattr(web, "_render_with_browser", lambda url: check)     # still blocked
    monkeypatch.setattr(web, "has_login", lambda: False)
    with pytest.raises(web.WebError, match="login.py"):
        web.fetch_page("https://www.zhipin.com/job_detail/1.html",
                       client=mock_client(lambda req: httpx.Response(200, text=check)))


def test_find_jobs_uses_a_long_search_summary(env, monkeypatch):
    from jobagent.schemas import JobCandidate, JobCandidates

    index, store = env
    monkeypatch.setattr(web, "search", lambda q, count=8, **kw: [
        {"title": "大模型评估专家", "url": "https://jobs/boss", "snippet": JD * 20, "date": ""},
        {"title": "短", "url": "https://jobs/short", "snippet": "岗位职责", "date": ""}])

    def blocked(url):
        raise web.WebError("The site showed a verification or login page instead of the job.")

    monkeypatch.setattr(web, "fetch_page", blocked)
    h = handlers("Co")
    h[SearchPlan] = lambda p: SearchPlan(queries=["q"])
    h[JobCandidates] = lambda p: JobCandidates(results=[
        JobCandidate(index=1, is_posting=True, is_open=True, relevant=True),
        JobCandidate(index=2, is_posting=True, is_open=True, relevant=True)])
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], h))}
    out = tools["find_jobs"].invoke({"request": "评测"})
    table = out.split("Notes:")[0]
    assert "https://jobs/boss" in table and "%*" in table and "search summary only" in table
    assert "https://jobs/short" not in table


def test_transient_errors_are_retried_and_auth_errors_are_not(monkeypatch):
    monkeypatch.setattr(web.config, "SEARCH_API_KEY", "k")
    monkeypatch.setattr(web.config, "SEARCH_PROVIDER", "bocha")
    calls = []

    def flaky(req):                                    # network drop, then rate limit, then success
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("dropped", request=req)
        if len(calls) == 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"data": {"webPages": {"value": [{"name": "t", "url": "https://x"}]}}})

    assert web.search("q", client=mock_client(flaky))[0]["url"] == "https://x" and len(calls) == 3

    calls.clear()
    unauthorized = mock_client(lambda req: calls.append(1) or httpx.Response(401))
    with pytest.raises(web.WebError, match="HTTP 401"):
        web.search("q", client=unauthorized)
    assert len(calls) == 1                             # a wrong key is not retried

    calls.clear()
    down = mock_client(lambda req: calls.append(1) or (_ for _ in ()).throw(httpx.ConnectError("x", request=req)))
    with pytest.raises(web.WebError, match="ConnectError"):
        web.search("q", client=down)
    assert len(calls) == web.config.HTTP_RETRIES       # gives up after the configured attempts

    calls.clear()
    page = mock_client(lambda req: calls.append(1) or (httpx.Response(503) if len(calls) < 2 else
                                                       httpx.Response(200, text=PAGE)))
    assert "岗位职责" in web.fetch_page("https://jobs/1", client=page) and len(calls) == 2
