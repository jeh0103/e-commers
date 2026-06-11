# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analytics import (
    customer_profile,
    customer_type_detail,
    dashboard_summary,
    generate_sms,
    risky_customers,
    vip_insights,
)
from app.data import (
    ROOT_DIR,
    apply_filters,
    clean_value,
    filter_options,
    format_kr_time,
    insert_action,
    label_for,
    load_actions,
    load_customers,
    records,
)


app = FastAPI(title="E-commerce Customer Ops", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT_DIR / "app" / "static"), name="static")

templates = Jinja2Templates(directory=ROOT_DIR / "app" / "templates")


def fmt_number(value: Any, digits: int = 0) -> str:
    value = clean_value(value)
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if digits <= 0:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def fmt_compact_number(value: Any, digits: int = 2) -> str:
    value = clean_value(value)
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) < 0.0000001:
        number = 0
    formatted = f"{number:,.{digits}f}"
    return formatted.rstrip("0").rstrip(".")


def fmt_detail_value(value: Any, label: str = "") -> str:
    label = str(label or "")
    value = clean_value(value)
    if value is None:
        return "-"
    if label in {"재구매 고객", "우수고객 여부", "리피트/프리미엄"}:
        try:
            return "예" if int(float(value)) == 1 else "아니오"
        except Exception:
            return str(value)
    if label == "나이":
        return fmt_number(value, 0)
    return fmt_compact_number(value, 2)


def fmt_percent(value: Any, digits: int = 1) -> str:
    value = clean_value(value)
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{digits}f}%"
    except Exception:
        return str(value)


templates.env.filters["num"] = fmt_number
templates.env.filters["detail_value"] = fmt_detail_value
templates.env.filters["pct"] = fmt_percent
templates.env.filters["label"] = label_for
templates.env.filters["krtime"] = format_kr_time


def status_label(value: Any) -> str:
    return {"open": "진행 중", "done": "완료", "hold": "보류"}.get(str(value or ""), str(value or ""))


templates.env.filters["status_label"] = status_label


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int_or_default(value: str | None, default: int, low: int | None = None, high: int | None = None) -> int:
    try:
        parsed = int(value) if value not in (None, "") else default
    except Exception:
        parsed = default
    if low is not None:
        parsed = max(low, parsed)
    if high is not None:
        parsed = min(high, parsed)
    return parsed


def parse_filter_query(request: Request) -> dict[str, Any]:
    qp = request.query_params
    premium = qp.get("premium", "all")
    if premium not in {"all", "yes", "no"}:
        premium = "all"
    return {
        "age_min": _float_or_none(qp.get("age_min")),
        "age_max": _float_or_none(qp.get("age_max")),
        "genders": qp.getlist("gender"),
        "premium": premium,
        "use_dynamic": qp.get("use_dynamic") in {"1", "true", "on", "yes"},
        "if_threshold": _float_or_none(qp.get("if_threshold")),
        "ae_threshold": _float_or_none(qp.get("ae_threshold")),
    }


def filtered_customers(request: Request) -> tuple[pd.DataFrame, str | None, dict[str, Any], dict[str, Any]]:
    df = load_customers()
    options = filter_options(df)
    filters = parse_filter_query(request)
    if filters["age_min"] is None:
        filters["age_min"] = options["age_min"]
    if filters["age_max"] is None:
        filters["age_max"] = options["age_max"]
    filtered, flag_col = apply_filters(df, **filters)
    return filtered, flag_col, filters, options


def base_context(request: Request, title: str) -> dict[str, Any]:
    return {
        "request": request,
        "title": title,
        "path": request.url.path,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    filtered, flag_col, filters, options = filtered_customers(request)
    summary = dashboard_summary(filtered, load_actions(), flag_col=flag_col)
    context = base_context(request, "고객 관리")
    context.update(
        {
            "filters": filters,
            "options": options,
            "summary": summary,
        }
    )
    return templates.TemplateResponse(request, "dashboard.html", context)


@app.get("/risky", response_class=HTMLResponse)
async def risky_page(request: Request) -> HTMLResponse:
    filtered, flag_col, filters, options = filtered_customers(request)
    src = request.query_params.get("src", "both")
    limit = _int_or_default(request.query_params.get("limit"), 100, 10, 500)
    result = risky_customers(filtered, src=src, limit=limit, fallback_flag_col=flag_col)
    context = base_context(request, result["config"]["title"])
    context.update(
        {
            "filters": filters,
            "options": options,
            "result": result,
            "limit": limit,
            "src": result["config"]["src"],
        }
    )
    return templates.TemplateResponse(request, "risky.html", context)


@app.get("/vip", response_class=HTMLResponse)
async def vip_page(request: Request) -> HTMLResponse:
    filtered, flag_col, filters, options = filtered_customers(request)
    qp = request.query_params
    result = vip_insights(
        filtered,
        mode=qp.get("mode", "threshold"),
        threshold=float(_float_or_none(qp.get("threshold")) or 80),
        topk=_int_or_default(qp.get("topk"), 100, 10, 1000),
        coverage_min_n=_int_or_default(qp.get("coverage_min_n"), 3, 1, 9),
        strong_signal_pct=float(_float_or_none(qp.get("strong_signal_pct")) or 95),
        clv_q=float(_float_or_none(qp.get("clv_q")) or 90),
        pf_q=float(_float_or_none(qp.get("pf_q")) or 80),
        logic=qp.get("logic", "and"),
        limit=_int_or_default(qp.get("limit"), 150, 10, 1000),
    )
    context = base_context(request, "VIP 관리")
    context.update(
        {
            "filters": filters,
            "options": options,
            "result": result,
            "params": {
                "mode": qp.get("mode", "threshold"),
                "threshold": float(_float_or_none(qp.get("threshold")) or 80),
                "topk": _int_or_default(qp.get("topk"), 100, 10, 1000),
                "coverage_min_n": _int_or_default(qp.get("coverage_min_n"), 3, 1, 9),
                "strong_signal_pct": float(_float_or_none(qp.get("strong_signal_pct")) or 95),
                "clv_q": float(_float_or_none(qp.get("clv_q")) or 90),
                "pf_q": float(_float_or_none(qp.get("pf_q")) or 80),
                "logic": qp.get("logic", "and"),
                "limit": _int_or_default(qp.get("limit"), 150, 10, 1000),
            },
        }
    )
    return templates.TemplateResponse(request, "vip.html", context)


@app.get("/customer-types", response_class=HTMLResponse)
async def customer_types_page(request: Request) -> HTMLResponse:
    filtered, flag_col, filters, options = filtered_customers(request)
    detail = customer_type_detail(
        filtered,
        load_actions(),
        customer_type=request.query_params.get("customer_type"),
        q=request.query_params.get("q", ""),
        only_high=request.query_params.get("only_high") in {"1", "on", "true"},
        only_no_contact=request.query_params.get("only_no_contact") in {"1", "on", "true"},
        min_risk=float(_float_or_none(request.query_params.get("min_risk")) or 0),
        flag_col=flag_col,
        limit=_int_or_default(request.query_params.get("limit"), 150, 10, 1000),
    )
    context = base_context(request, "고객 유형")
    context.update(
        {
            "filters": filters,
            "options": options,
            "detail": detail,
            "query": {
                "q": request.query_params.get("q", ""),
                "only_high": request.query_params.get("only_high") in {"1", "on", "true"},
                "only_no_contact": request.query_params.get("only_no_contact") in {"1", "on", "true"},
                "min_risk": float(_float_or_none(request.query_params.get("min_risk")) or 0),
                "limit": _int_or_default(request.query_params.get("limit"), 150, 10, 1000),
            },
        }
    )
    return templates.TemplateResponse(request, "customer_types.html", context)


@app.get("/customers/{customer_id}", response_class=HTMLResponse)
async def customer_detail_page(request: Request, customer_id: str) -> HTMLResponse:
    profile = customer_profile(load_customers(), load_actions(), customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="고객을 찾을 수 없습니다.")
    context = base_context(request, f"고객 상세 {customer_id}")
    context.update({"customer_id": customer_id, "profile": profile})
    return templates.TemplateResponse(request, "customer_detail.html", context)


@app.get("/api/summary")
async def api_summary(request: Request) -> JSONResponse:
    filtered, flag_col, _, _ = filtered_customers(request)
    return JSONResponse(dashboard_summary(filtered, load_actions(), flag_col=flag_col))


@app.get("/api/customers")
async def api_customers(request: Request) -> JSONResponse:
    filtered, _, _, _ = filtered_customers(request)
    q = request.query_params.get("q", "").strip()
    if q:
        filtered = filtered[filtered["CustomerID_clean"].astype(str).str.contains(q, case=False, na=False)]
    sort = request.query_params.get("sort", "RiskScore100")
    direction = request.query_params.get("direction", "desc")
    if sort in filtered.columns:
        filtered = filtered.sort_values(sort, ascending=(direction == "asc"), na_position="last")
    limit = _int_or_default(request.query_params.get("limit"), 100, 1, 1000)
    columns = [
        "CustomerID_clean",
        "GenderLabel",
        "Age",
        "CustomerType",
        "RiskLevel",
        "RiskScore100",
        "PurchaseFrequency",
        "CSFrequency",
        "AverageSatisfactionScore",
        "TotalEngagementScore",
    ]
    return JSONResponse({"total": int(len(filtered)), "rows": records(filtered, columns=columns, limit=limit)})


@app.get("/api/customers/{customer_id}")
async def api_customer(customer_id: str) -> JSONResponse:
    profile = customer_profile(load_customers(), load_actions(), customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="고객을 찾을 수 없습니다.")
    return JSONResponse(profile)


@app.post("/api/customers/{customer_id}/actions")
async def api_create_action(customer_id: str, request: Request) -> JSONResponse:
    payload = await request.json()
    action = str(payload.get("action") or "").strip()
    if not action:
        raise HTTPException(status_code=422, detail="action 값이 필요합니다.")
    owner = str(payload.get("owner") or "").strip()
    status = str(payload.get("status") or "open").strip()
    if status not in {"open", "done", "hold"}:
        status = "open"
    note = str(payload.get("note") or "").strip()
    action_id = insert_action(customer_id, action=action, note=note, owner=owner, status=status)
    return JSONResponse({"ok": True, "id": action_id})


@app.post("/api/customers/{customer_id}/sms-preview")
async def api_sms_preview(customer_id: str, request: Request) -> JSONResponse:
    df = load_customers()
    profile = customer_profile(df, load_actions(), customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="고객을 찾을 수 없습니다.")
    payload = await request.json()
    result = generate_sms(
        customer_id,
        brand=str(payload.get("brand") or "브랜드"),
        benefit=str(payload.get("benefit") or "5% 할인 쿠폰"),
        expiry=payload.get("expiry") or None,
        tone=str(payload.get("tone") or "정중"),
        theme=str(payload.get("theme") or "auto"),
        landing_url=str(payload.get("landing_url") or ""),
        cs_contact=str(payload.get("cs_contact") or ""),
        optout=str(payload.get("optout") or "수신거부: 수신중지"),
        target_segments=int(payload.get("target_segments") or 1),
        drivers=profile["drivers"],
    )
    return JSONResponse(result)


def csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    data = ("\ufeff" + stream.getvalue()).encode("utf-8")
    return StreamingResponse(
        iter([data]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/risky.csv")
async def api_risky_csv(request: Request) -> StreamingResponse:
    filtered, flag_col, _, _ = filtered_customers(request)
    src = request.query_params.get("src", "both")
    limit = _int_or_default(request.query_params.get("limit"), 500, 1, 5000)
    result = risky_customers(filtered, src=src, limit=limit, fallback_flag_col=flag_col)
    return csv_response(pd.DataFrame(result["rows"]), f"{result['config']['src']}_risky_customers.csv")


@app.get("/api/vip-candidates.csv")
async def api_vip_candidates_csv(request: Request) -> StreamingResponse:
    filtered, _, _, _ = filtered_customers(request)
    result = vip_insights(filtered, limit=_int_or_default(request.query_params.get("limit"), 1000, 1, 5000))
    return csv_response(result["candidate_df"], "vip_candidates.csv")
