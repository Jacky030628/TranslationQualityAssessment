import os
import re
import json
import uuid
import time
from pathlib import Path
from typing import List, Literal, Dict, Any

import requests
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from openpyxl import Workbook


# ========= 基础配置 =========
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_CHAT_COMPLETIONS_URL = f"{DEEPSEEK_BASE_URL}/chat/completions"
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Translation QA (DeepSeek) Prototype", version="0.3.0")
templates = Jinja2Templates(directory="templates")


# ========= 新 Rubric 数据结构（House functional-pragmatic） =========
Severity3 = Literal["Critical", "Major", "Minor"]
ErrorType = Literal["Overt", "Covert"]


class RubricDimension(BaseModel):
    code: Literal["A", "B", "C", "D", "E"]
    name: str
    max_score: int


class Rubric(BaseModel):
    name: str
    dimensions: List[RubricDimension]


class RubricIssue(BaseModel):
    severity: Severity3
    error_type: ErrorType
    delta: int = Field(ge=0)
    evidence_source: str = ""
    evidence_target: str
    explanation: str
    suggestion: str


class DimensionResult(BaseModel):
    code: Literal["A", "B", "C", "D", "E"]
    name: str
    score: int = Field(ge=0)
    max_score: int = Field(ge=0)
    rationale: str = ""
    issues: List[RubricIssue] = Field(default_factory=list)


class RubricResult(BaseModel):
    rubric: Rubric
    dimensions: List[DimensionResult]
    total_score: int = Field(ge=0, le=100)
    overall_summary: str


# ========= Prompt 模板 =========
PROFILE_HINTS = {
    "academic": "学术写作：准确、客观、逻辑清晰，避免口语化。",
    "business": "商务写作：简洁专业，强调可读性与一致性。",
    "news": "新闻写作：准确清楚，避免夸张修饰，语体适中。",
    "general": "通用：优先准确，其次地道性和表达自然。",
}

PROMPT_TEMPLATE = """你是翻译质量评估与审校助手。请严格依据下列“House 风格的功能—语用等值 Rubric”对【译文】相对于【原文】进行评分与诊断。评分总分为 100 分。

【风格预设（辅助约束）】
{profile_hint}

【评估维度与权重（满分=100）】
A. 语义准确 Semantic Accuracy（35）
- 核心问题：关键信息是否理解/传达正确？有无错译/漏译/增译导致歪曲？
- 对应：semantic meaning / overt errors

B. 语用得体 Pragmatic Appropriateness（15）
- 核心问题：语气、态度、礼貌、言外之意、立场是否匹配原文语境？
- 对应：pragmatic meaning / covert errors

C. 篇章与逻辑 Textual Coherence（15）
- 核心问题：衔接、连贯、指代、信息结构、逻辑关系是否清晰且一致？
- 对应：textual meaning

D. 语域匹配 Register Match（20）
- 核心问题：Field/Tenor/Mode 是否与原文一致（内容活动、关系态度、媒介复杂度/连接性）？
- 对应：register: field/tenor/mode

E. 语言表达 Fluency & Style（15）
- 核心问题：目标语是否自然、语法/搭配/标点规范，符合该语类（Genre）常规？
- 对应：genre + realization

【错误类型（用于解释，不直接等同扣分）】
- Overt errors（显性错误）：误解/错译/漏译/语法导致意义错误/专名术语错误/数字单位错误
- Covert errors（隐性错误）：语域不当/文体不匹配/语用失衡（过口语或过正式、态度变化）/不符合语类常规

【严重度分级（驱动扣分与解释）】
- Critical：改变核心事实/逻辑/立场；可能导致严重误用（法律、医疗、金额、否定关系等）
- Major：影响理解或明显失真，但不至于完全反向
- Minor：不影响主旨，主要是自然度/措辞/格式问题

【扣分原则（请严格执行）】
1) 每个维度分别从其满分开始扣分，最低为 0。
2) 扣分应与“严重度”一致，并在 issue 中给出 delta（扣分值）。
   - 参考范围（可根据文本长度与影响微调，但要自洽）：
     Critical：该维度扣 6–12 分
     Major：该维度扣 3–6 分
     Minor：该维度扣 1–3 分
3) 每个扣分点必须提供证据：
   - evidence_source：对应原文片段（必要时可为空，但尽量给）
   - evidence_target：对应译文片段（必须给）
4) 每个扣分点必须给出可执行的修改建议 suggestion（尽量给“可直接替换”的译文写法）。
5) 最终 total_score = A_score + B_score + C_score + D_score + E_score（必须等于 0–100 的整数）。

【输出要求（非常重要）】
- 只输出“有效 JSON”，不要输出任何多余文字、不要 Markdown、不要解释。
- 字段名必须完全一致，缺省字段也必须存在。
- issues 可以为空数组，但 dimensions 必须包含 A–E 五项。
- 每个维度都必须给 rationale（哪怕 issues 为空）。

【必须输出的 JSON 结构】
{{
  "rubric": {{
    "name": "House functional-pragmatic equivalence (custom)",
    "dimensions": [
      {{"code":"A","name":"Semantic Accuracy","max_score":35}},
      {{"code":"B","name":"Pragmatic Appropriateness","max_score":15}},
      {{"code":"C","name":"Textual Coherence","max_score":15}},
      {{"code":"D","name":"Register Match","max_score":20}},
      {{"code":"E","name":"Fluency & Style","max_score":15}}
    ]
  }},
  "dimensions": [
    {{
      "code": "A",
      "name": "Semantic Accuracy",
      "score": 0,
      "max_score": 35,
      "rationale": "",
      "issues": [
        {{
          "severity": "Critical|Major|Minor",
          "error_type": "Overt|Covert",
          "delta": 0,
          "evidence_source": "",
          "evidence_target": "",
          "explanation": "",
          "suggestion": ""
        }}
      ]
    }}
  ],
  "total_score": 0,
  "overall_summary": ""
}}

【原文】
{source}

【译文】
{target}
"""


# ========= 工具函数 =========
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _get_deepseek_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY 未设置。请先设置环境变量。")
    return key


def call_deepseek_chat(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2, max_tokens: int = 1600) -> str:
    api_key = _get_deepseek_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only generator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(DEEPSEEK_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        if resp.status_code == 402 and "Insufficient Balance" in resp.text:
            raise HTTPException(status_code=402, detail="Insufficient Balance（余额不足）")
        raise HTTPException(status_code=502, detail=f"DeepSeek API 调用失败：{resp.status_code} {resp.text}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"DeepSeek 返回结构异常：{data}")


def _rubric_sanity_check(result: RubricResult) -> None:
    # 1) 维度必须 A-E 且唯一
    codes = [d.code for d in result.dimensions]
    if sorted(codes) != ["A", "B", "C", "D", "E"]:
        raise ValueError(f"dimensions 必须且只能包含 A-E 五项，当前：{codes}")

    # 2) total_score 必须等于维度分数之和
    s = sum(d.score for d in result.dimensions)
    if s != result.total_score:
        raise ValueError(f"total_score 不等于维度分数之和：sum={s}, total_score={result.total_score}")

    # 3) 每个维度 score 不超过 max_score，且 max_score 与 rubric 要一致（尽量）
    rubric_map = {d.code: d.max_score for d in result.rubric.dimensions}
    for d in result.dimensions:
        if d.score > d.max_score:
            raise ValueError(f"{d.code} 维度 score({d.score}) > max_score({d.max_score})")
        if d.code in rubric_map and d.max_score != rubric_map[d.code]:
            # 不强制失败也行，但这里选择严格一点，保证一致性
            raise ValueError(f"{d.code} 维度 max_score({d.max_score}) 与 rubric({rubric_map[d.code]}) 不一致")


def diagnose_once(source: str, target: str, profile: str) -> RubricResult:
    profile_hint = PROFILE_HINTS.get(profile, PROFILE_HINTS["general"])
    prompt = PROMPT_TEMPLATE.format(
        profile_hint=profile_hint,
        source=source.strip(),
        target=target.strip(),
    )

    raw = call_deepseek_chat(prompt)
    obj = _safe_json_loads(raw)

    try:
        result = RubricResult.model_validate(obj)
    except ValidationError as e:
        raise ValueError(f"JSON schema 校验失败：{e}")

    _rubric_sanity_check(result)
    return result


def diagnose_with_retry(source: str, target: str, profile: str, retries: int = 1) -> RubricResult:
    last_err = None
    for attempt in range(retries + 1):
        try:
            return diagnose_once(source, target, profile)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    raise HTTPException(status_code=500, detail=f"诊断失败（多次尝试后仍失败）：{last_err}")


def save_report(report_id: str, source: str, target: str, profile: str, result: RubricResult) -> Path:
    fp = REPORTS_DIR / f"{report_id}.json"
    payload = {
        "id": report_id,
        "profile": profile,
        "source": source,
        "target": target,
        "result": result.model_dump(),
        "created_at": int(time.time()),
        "model": DEFAULT_MODEL,
        "base_url": DEEPSEEK_BASE_URL,
    }
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def load_report(report_id: str) -> Dict[str, Any]:
    fp = REPORTS_DIR / f"{report_id}.json"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    return json.loads(fp.read_text(encoding="utf-8"))


def export_report_to_xlsx(report: Dict[str, Any]) -> Path:
    rid = report["id"]
    result = report["result"]

    # summary sheet
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "summary"

    # 维度列
    dim_scores = {}
    for d in result.get("dimensions", []):
        dim_scores[d.get("code")] = f'{d.get("score")}/{d.get("max_score")}'

    ws1.append([
        "id", "profile", "total_score", "overall_summary",
        "A", "B", "C", "D", "E",
        "issues_total",
        "model", "base_url"
    ])
    issues_total = 0
    for d in result.get("dimensions", []):
        issues_total += len(d.get("issues", []))

    ws1.append([
        rid,
        report.get("profile", ""),
        result.get("total_score", ""),
        result.get("overall_summary", ""),
        dim_scores.get("A", ""),
        dim_scores.get("B", ""),
        dim_scores.get("C", ""),
        dim_scores.get("D", ""),
        dim_scores.get("E", ""),
        issues_total,
        report.get("model", ""),
        report.get("base_url", ""),
    ])

    # details sheet
    ws2 = wb.create_sheet("issues")
    ws2.append([
        "id", "dimension_code", "dimension_name",
        "severity", "error_type", "delta",
        "evidence_source", "evidence_target",
        "explanation", "suggestion"
    ])

    for d in result.get("dimensions", []):
        for it in d.get("issues", []):
            ws2.append([
                rid,
                d.get("code", ""),
                d.get("name", ""),
                it.get("severity", ""),
                it.get("error_type", ""),
                it.get("delta", ""),
                it.get("evidence_source", ""),
                it.get("evidence_target", ""),
                it.get("explanation", ""),
                it.get("suggestion", ""),
            ])

    out_fp = REPORTS_DIR / f"{rid}.xlsx"
    wb.save(out_fp)
    return out_fp


# ========= 路由：网页 =========
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_profile": "general"})


@app.get("/report/{report_id}", response_class=HTMLResponse)
def report_page(request: Request, report_id: str):
    report = load_report(report_id)
    return templates.TemplateResponse("report.html", {"request": request, "report": report})


@app.post("/diagnose")
def web_diagnose(
    source: str = Form(...),
    target: str = Form(...),
    profile: str = Form("general"),
):
    if not source.strip() or not target.strip():
        raise HTTPException(status_code=400, detail="原文/译文不能为空")

    report_id = uuid.uuid4().hex[:12]
    result = diagnose_with_retry(source, target, profile, retries=1)
    save_report(report_id, source, target, profile, result)

    return RedirectResponse(url=f"/report/{report_id}", status_code=303)


# ========= 路由：API（给以后做 Ajax 或对接别的前端用） =========
@app.post("/api/diagnose")
def api_diagnose(
    source: str = Form(...),
    target: str = Form(...),
    profile: str = Form("general"),
):
    if not source.strip() or not target.strip():
        raise HTTPException(status_code=400, detail="原文/译文不能为空")

    report_id = uuid.uuid4().hex[:12]
    result = diagnose_with_retry(source, target, profile, retries=1)
    save_report(report_id, source, target, profile, result)

    dim_map = {d.code: d.score for d in result.dimensions}
    issue_total = sum(len(d.issues) for d in result.dimensions)

    return JSONResponse({
        "report_id": report_id,
        "total_score": result.total_score,
        "overall_summary": result.overall_summary,
        "dimension_scores": dim_map,
        "issues_total": issue_total,
        "redirect": f"/report/{report_id}"
    })


@app.get("/api/report/{report_id}")
def api_get_report(report_id: str):
    return JSONResponse(load_report(report_id))


@app.get("/api/export/{report_id}.xlsx")
def api_export_xlsx(report_id: str):
    report = load_report(report_id)
    fp = export_report_to_xlsx(report)
    return FileResponse(
        path=str(fp),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{report_id}.xlsx",
    )
