# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd

from app.data import (
    ACTIONS_LOOKBACK_DAYS,
    clean_value,
    label_for,
    recent_action_sets,
    records,
    risk_level,
)
from utils_vip import (
    backtest_metrics,
    compute_vip_propensity_score,
    roi_for_k,
    select_vip_candidates,
)


RISK_DRIVER_DIR = {
    "CSFrequency": "higher_worse",
    "RecencyProxy": "higher_worse",
    "NegativeExperienceIndex": "higher_worse",
    "AvgPurchaseInterval": "higher_worse",
    "PurchaseFrequency": "lower_worse",
    "AverageSatisfactionScore": "lower_worse",
    "EmailEngagementRate": "lower_worse",
    "TotalEngagementScore": "lower_worse",
    "TotalPurchases": "lower_worse",
    "AverageOrderValue": "neutral",
}

RISK_DRIVER_NAMES = {
    "PurchaseFrequency": "구매 빈도",
    "CSFrequency": "상담 빈도",
    "RecencyProxy": "활동저하 지수",
    "AverageSatisfactionScore": "만족도",
    "NegativeExperienceIndex": "부정 경험 지수",
    "EmailEngagementRate": "이메일 참여율",
    "TotalEngagementScore": "총 참여 점수",
    "AvgPurchaseInterval": "평균 구매 간격",
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    s = _numeric(series).dropna()
    return float(s.mean()) if len(s) else 0.0


def _quantile(series: pd.Series, q: float) -> float | None:
    s = _numeric(series).dropna()
    return float(s.quantile(q)) if len(s) else None


def attach_vip_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = compute_vip_propensity_score(df, ref_df=df).reset_index(drop=True)
    out = df.reset_index(drop=True).copy()
    for col in scored.columns:
        if col in out.columns:
            continue
        out[col] = scored[col]
    return out


def source_config(src: str, df: pd.DataFrame, fallback_flag_col: str | None = None) -> dict[str, Any]:
    src = (src or "both").lower()
    if src == "if":
        flag = "IF_ChurnFlag_dyn" if "IF_ChurnFlag_dyn" in df.columns else "IF_ChurnFlag"
        return {
            "src": "if",
            "title": "이상 행동 고객",
            "flag_col": flag if flag in df.columns else fallback_flag_col,
            "sort_col": "IF_AnomalyScore" if "IF_AnomalyScore" in df.columns else "ChurnRiskScore",
            "metric_label": "이상 행동 점수",
        }
    if src == "ae":
        flag = "AE_ChurnFlag_dyn" if "AE_ChurnFlag_dyn" in df.columns else "AE_ChurnFlag"
        return {
            "src": "ae",
            "title": "패턴 변화 고객",
            "flag_col": flag if flag in df.columns else fallback_flag_col,
            "sort_col": "AE_ReconError" if "AE_ReconError" in df.columns else "ChurnRiskScore",
            "metric_label": "패턴 변화 점수",
        }
    return {
        "src": "both",
        "title": "우선 관리 고객",
        "flag_col": fallback_flag_col if fallback_flag_col in df.columns else ("Both_ChurnFlag" if "Both_ChurnFlag" in df.columns else None),
        "sort_col": "ChurnRiskScore" if "ChurnRiskScore" in df.columns else "RiskScore100",
        "metric_label": "이탈 위험 점수",
    }


def risk_tags(row: pd.Series, ref_df: pd.DataFrame) -> list[str]:
    checks = [
        ("NegativeExperienceIndex", 0.80, "high", "부정경험 높음"),
        ("AverageSatisfactionScore", 0.20, "low", "만족도 낮음"),
        ("EmailEngagementRate", 0.20, "low", "이메일 참여 낮음"),
        ("CSFrequency", 0.80, "high", "상담 빈도 높음"),
        ("TotalEngagementScore", 0.20, "low", "참여 점수 낮음"),
        ("AvgPurchaseInterval", 0.80, "high", "구매 간격 김"),
        ("PurchaseFrequency", 0.20, "low", "구매 빈도 낮음"),
    ]
    tags: list[str] = []
    for col, q, direction, label in checks:
        if col not in ref_df.columns or col not in row.index or pd.isna(row[col]):
            continue
        threshold = _quantile(ref_df[col], q)
        if threshold is None:
            continue
        value = float(row[col])
        if (direction == "high" and value >= threshold) or (direction == "low" and value <= threshold):
            tags.append(label)
    return tags[:4]


def priority_index(ref_series: pd.Series, values: pd.Series) -> pd.Series:
    ref = _numeric(ref_series).dropna()
    val = _numeric(values)
    if ref.empty:
        return pd.Series(0, index=values.index)
    lo = float(ref.quantile(0.05))
    hi = float(ref.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(ref.min()), float(ref.max())
    rng = hi - lo if hi > lo else 1.0
    return (((val - lo) / rng).clip(0, 1) * 100).round(0).fillna(0)


def priority_label(score: Any) -> str:
    try:
        value = float(score)
    except Exception:
        return "후순위"
    if value >= 90:
        return "최우선"
    if value >= 70:
        return "높음"
    if value >= 40:
        return "보통"
    return "후순위"


def risky_customers(df: pd.DataFrame, src: str = "both", limit: int = 100, fallback_flag_col: str | None = None) -> dict[str, Any]:
    cfg = source_config(src, df, fallback_flag_col=fallback_flag_col)
    subset = df.copy()
    flag_col = cfg["flag_col"]
    if flag_col and flag_col in subset.columns:
        subset = subset[pd.to_numeric(subset[flag_col], errors="coerce").fillna(0).astype(int) == 1]
    if "CustomerID_clean" in subset.columns:
        subset = subset[subset["CustomerID_clean"].notna()]

    sort_col = cfg["sort_col"]
    if sort_col in subset.columns:
        subset = subset.sort_values(sort_col, ascending=False, na_position="last")
    elif "RiskScore100" in subset.columns:
        subset = subset.sort_values("RiskScore100", ascending=False, na_position="last")

    if sort_col in df.columns and sort_col in subset.columns:
        subset["PriorityIndex"] = priority_index(df[sort_col], subset[sort_col])
    else:
        subset["PriorityIndex"] = subset.get("RiskScore100", pd.Series(0, index=subset.index))
    subset["PriorityLabel"] = subset["PriorityIndex"].map(priority_label)
    subset["RiskTags"] = [risk_tags(row, df) for _, row in subset.iterrows()]

    columns = [
        "CustomerID_clean",
        "GenderLabel",
        "Age",
        "CustomerType",
        "RiskLevel",
        "RiskScore100",
        "PriorityIndex",
        "PriorityLabel",
        "RiskTags",
        "PurchaseFrequency",
        "CSFrequency",
        "AverageSatisfactionScore",
        "NegativeExperienceIndex",
        "EmailEngagementRate",
        "TotalEngagementScore",
        sort_col,
    ]
    rows = records(subset, columns=columns, limit=limit)
    return {
        "config": cfg,
        "total": int(len(subset)),
        "rows": rows,
        "columns": [c for c in columns if c in subset.columns or c in {"PriorityIndex", "PriorityLabel", "RiskTags"}],
    }


def customer_type_distribution(df: pd.DataFrame, flag_col: str | None = None) -> list[dict[str, Any]]:
    if "CustomerType" not in df.columns:
        return []
    work = df.copy()
    if flag_col and flag_col in work.columns:
        work["_high_churn"] = pd.to_numeric(work[flag_col], errors="coerce").fillna(0).astype(int)
    else:
        work["_high_churn"] = 0

    dist = (
        work.groupby("CustomerType", dropna=False)
        .agg(
            customer_count=("CustomerType", "size"),
            avg_risk=("RiskScore100", "mean"),
            high_churn_rate=("_high_churn", "mean"),
        )
        .reset_index()
    )
    dist["share"] = dist["customer_count"] / max(1, len(work)) * 100
    dist["high_churn_rate"] = dist["high_churn_rate"] * 100
    dist = dist.sort_values(["high_churn_rate", "customer_count"], ascending=[False, False])
    return records(dist)


def customer_type_detail(
    df: pd.DataFrame,
    actions: pd.DataFrame,
    customer_type: str | None,
    q: str = "",
    only_high: bool = False,
    only_no_contact: bool = False,
    min_risk: float = 0,
    flag_col: str | None = None,
    limit: int = 300,
) -> dict[str, Any]:
    all_types = sorted(df["CustomerType"].dropna().astype(str).unique().tolist()) if "CustomerType" in df.columns else ["미분류"]
    selected = customer_type if customer_type in all_types else (all_types[0] if all_types else "미분류")
    work = df[df["CustomerType"] == selected].copy()

    contacted, benefit = recent_action_sets(actions)
    work["RecentContact"] = work["CustomerID_clean"].astype(str).isin(contacted)
    work["RecentBenefit"] = work["CustomerID_clean"].astype(str).isin(benefit)
    if flag_col and flag_col in work.columns:
        work["HighChurn"] = pd.to_numeric(work[flag_col], errors="coerce").fillna(0).astype(int) == 1
    else:
        work["HighChurn"] = False

    if q:
        work = work[work["CustomerID_clean"].astype(str).str.contains(q, case=False, na=False)]
    if only_high:
        work = work[work["HighChurn"]]
    if only_no_contact:
        work = work[~work["RecentContact"]]
    work = work[pd.to_numeric(work["RiskScore100"], errors="coerce").fillna(0) >= float(min_risk)]

    if "RiskScore100" in work.columns:
        work = work.sort_values("RiskScore100", ascending=False, na_position="last")

    metrics = {
        "customer_count": int(len(df[df["CustomerType"] == selected])),
        "filtered_count": int(len(work)),
        "high_churn_rate": float(work["HighChurn"].mean() * 100) if len(work) else 0.0,
        "avg_risk": _safe_mean(work["RiskScore100"]) if len(work) else 0.0,
        "no_contact_count": int((~work["RecentContact"]).sum()) if len(work) else 0,
    }

    feature_rows = []
    for col, label, direction in [
        ("PurchaseFrequency", "구매 빈도", "낮을수록 위험"),
        ("CSFrequency", "상담 빈도", "높을수록 위험"),
        ("AverageSatisfactionScore", "평균 만족도", "낮을수록 위험"),
        ("NegativeExperienceIndex", "부정 경험 지수", "높을수록 위험"),
        ("TotalEngagementScore", "총 참여 점수", "낮을수록 위험"),
    ]:
        if col not in df.columns or col not in work.columns:
            continue
        overall = _safe_mean(df[col])
        current = _safe_mean(work[col])
        if overall:
            delta = (current - overall) / abs(overall) * 100
        else:
            delta = current - overall
        feature_rows.append(
            {
                "label": label,
                "current": current,
                "overall": overall,
                "delta": delta,
                "direction": direction,
            }
        )
    feature_rows = sorted(feature_rows, key=lambda x: abs(x["delta"]), reverse=True)

    columns = [
        "CustomerID_clean",
        "GenderLabel",
        "Age",
        "RepeatAndPremiumFlag",
        "RecentContact",
        "RecentBenefit",
        "HighChurn",
        "RiskLevel",
        "RiskScore100",
        "PurchaseFrequency",
        "CSFrequency",
        "AverageSatisfactionScore",
    ]
    return {
        "types": all_types,
        "selected": selected,
        "metrics": metrics,
        "features": feature_rows[:3],
        "rows": records(work, columns=columns, limit=limit),
    }


def dashboard_summary(df: pd.DataFrame, actions: pd.DataFrame, flag_col: str | None = None) -> dict[str, Any]:
    total = int(len(df))
    churn_if_col = "IF_ChurnFlag_dyn" if "IF_ChurnFlag_dyn" in df.columns else "IF_ChurnFlag"
    churn_ae_col = "AE_ChurnFlag_dyn" if "AE_ChurnFlag_dyn" in df.columns else "AE_ChurnFlag"
    churn_if = int(pd.to_numeric(df[churn_if_col], errors="coerce").fillna(0).sum()) if churn_if_col in df.columns else 0
    churn_ae = int(pd.to_numeric(df[churn_ae_col], errors="coerce").fillna(0).sum()) if churn_ae_col in df.columns else 0
    churn_both = int(pd.to_numeric(df[flag_col], errors="coerce").fillna(0).sum()) if flag_col and flag_col in df.columns else 0

    contacted, benefit = recent_action_sets(actions)

    risky = df.copy()
    if flag_col and flag_col in risky.columns:
        risky = risky[pd.to_numeric(risky[flag_col], errors="coerce").fillna(0).astype(int) == 1]
    risky = risky[~risky["CustomerID_clean"].astype(str).isin(contacted)]
    if "ChurnRiskScore" in risky.columns:
        risky = risky.sort_values("ChurnRiskScore", ascending=False, na_position="last")
    risky["RiskTags"] = [risk_tags(row, df) for _, row in risky.iterrows()]

    vip_no_benefit = pd.DataFrame()
    try:
        tmp = attach_vip_scores(df)
        vip_no_benefit = tmp[
            (_numeric(tmp["VIP잠재지수"]) >= 80)
            & (~tmp["CustomerID_clean"].astype(str).isin(benefit))
        ].sort_values("VIP잠재지수", ascending=False, na_position="last")
    except Exception:
        vip_no_benefit = pd.DataFrame()

    open_actions = 0
    if not actions.empty and "status" in actions.columns:
        open_actions = int(actions["status"].fillna("").astype(str).isin({"open", "hold"}).sum())

    return {
        "metrics": {
            "total": total,
            "churn_if": churn_if,
            "churn_ae": churn_ae,
            "churn_both": churn_both,
            "churn_both_rate": (churn_both / total * 100) if total else 0.0,
            "risky_no_contact": int(len(risky)),
            "vip_no_benefit": int(len(vip_no_benefit)),
            "open_actions": open_actions,
        },
        "risky_priority": records(
            risky,
            columns=["CustomerID_clean", "RiskLevel", "RiskScore100", "RiskTags", "PurchaseFrequency", "CSFrequency"],
            limit=10,
        ),
        "vip_priority": records(
            vip_no_benefit,
            columns=["CustomerID_clean", "VIP잠재지수", "CustomerLifetimeValue", "PurchaseFrequency", "AverageOrderValue", "TotalEngagementScore"],
            limit=7,
        ),
        "customer_types": customer_type_distribution(df, flag_col=flag_col),
        "risky_preview": risky_customers(df, src="both", limit=12, fallback_flag_col=flag_col)["rows"],
    }


def recommend_tags(row: pd.Series, ref_df: pd.DataFrame) -> str:
    tags: list[str] = []
    if "AverageOrderValue" in row and "AverageOrderValue" in ref_df.columns:
        thr = _quantile(ref_df["AverageOrderValue"], 0.85)
        if thr is not None and pd.notna(row["AverageOrderValue"]) and row["AverageOrderValue"] >= thr:
            tags.append("프리미엄/한정판")
    if "PurchaseFrequency" in row and "PurchaseFrequency" in ref_df.columns:
        thr = _quantile(ref_df["PurchaseFrequency"], 0.85)
        if thr is not None and pd.notna(row["PurchaseFrequency"]) and row["PurchaseFrequency"] >= thr:
            tags.append("멤버십 상향")
    if "TotalEngagementScore" in row and "TotalEngagementScore" in ref_df.columns:
        thr = _quantile(ref_df["TotalEngagementScore"], 0.80)
        if thr is not None and pd.notna(row["TotalEngagementScore"]) and row["TotalEngagementScore"] >= thr:
            tags.append("신상품 우선 안내")
    if "EmailEngagementRate" in row and "EmailEngagementRate" in ref_df.columns:
        thr = _quantile(ref_df["EmailEngagementRate"], 0.70)
        if thr is not None and pd.notna(row["EmailEngagementRate"]) and row["EmailEngagementRate"] >= thr:
            tags.append("개인화 쿠폰")
    if "MobileAppUsage" in row and "MobileAppUsage" in ref_df.columns:
        thr = _quantile(ref_df["MobileAppUsage"], 0.30)
        if thr is not None and pd.notna(row["MobileAppUsage"]) and row["MobileAppUsage"] < thr:
            tags.append("앱 온보딩")
    if "AvgPurchaseInterval" in row and "AvgPurchaseInterval" in ref_df.columns:
        thr = _quantile(ref_df["AvgPurchaseInterval"], 0.80)
        if thr is not None and pd.notna(row["AvgPurchaseInterval"]) and row["AvgPurchaseInterval"] >= thr:
            tags.append("재구매 리마인드")
    return " / ".join(tags or ["VIP 전용 상담 / 무료반품 / 생일쿠폰"])


def vip_insights(
    df: pd.DataFrame,
    mode: str = "threshold",
    threshold: float = 80,
    topk: int = 100,
    coverage_min_n: int = 3,
    strong_signal_pct: float = 95,
    clv_q: float = 90,
    pf_q: float = 80,
    logic: str = "and",
    limit: int = 150,
) -> dict[str, Any]:
    scored_full = attach_vip_scores(df)

    candidates, snapshot = select_vip_candidates(
        scored_full,
        mode="topk" if mode == "topk" else "threshold",
        k=int(topk),
        thr=float(threshold),
        coverage_min_n=int(coverage_min_n),
        strong_signal_pct=float(strong_signal_pct),
    )
    candidates = candidates.sort_values("VIP잠재지수", ascending=False, na_position="last")
    candidates["Recommendation"] = [recommend_tags(row, df) for _, row in candidates.iterrows()]
    candidates["CoverageLabel"] = pd.to_numeric(candidates.get("coverage", 0), errors="coerce").fillna(0).map(
        lambda v: "신뢰도 높음" if v >= 0.75 else ("신뢰도 보통" if v >= 0.45 else "신뢰도 낮음")
    )

    clv_cut = _quantile(df["CustomerLifetimeValue"], clv_q / 100.0) if "CustomerLifetimeValue" in df.columns else None
    pf_cut = _quantile(df["PurchaseFrequency"], pf_q / 100.0) if "PurchaseFrequency" in df.columns else None
    mask_clv = _numeric(df["CustomerLifetimeValue"]) >= (clv_cut if clv_cut is not None else np.inf) if "CustomerLifetimeValue" in df.columns else False
    mask_pf = _numeric(df["PurchaseFrequency"]) >= (pf_cut if pf_cut is not None else np.inf) if "PurchaseFrequency" in df.columns else False
    vip_mask = (mask_clv & mask_pf) if logic == "and" else (mask_clv | mask_pf)
    current_vip = df[vip_mask].copy()
    current_vip = current_vip.sort_values("CustomerLifetimeValue", ascending=False, na_position="last") if "CustomerLifetimeValue" in current_vip.columns else current_vip
    current_vip["Recommendation"] = [recommend_tags(row, df) for _, row in current_vip.iterrows()]

    k_eval = min(100, max(1, len(scored_full) // 20))
    bt = backtest_metrics(scored_full, score_col="VIP잠재지수", label_col=None, k=k_eval)
    roi = roi_for_k(scored_full, k=min(100, len(scored_full)), avg_order_value=50000, gross_margin=0.35, cost_per_contact=1000)

    columns = [
        "CustomerID_clean",
        "VIP잠재지수",
        "CoverageLabel",
        "coverage",
        "CustomerLifetimeValue",
        "PurchaseFrequency",
        "AverageOrderValue",
        "TotalEngagementScore",
        "EmailEngagementRate",
        "MobileAppUsage",
        "AvgPurchaseInterval",
        "Recommendation",
    ]
    vip_columns = [
        "CustomerID_clean",
        "CustomerLifetimeValue",
        "PurchaseFrequency",
        "AverageOrderValue",
        "TotalEngagementScore",
        "EmailEngagementRate",
        "MobileAppUsage",
        "Recommendation",
    ]
    return {
        "snapshot": snapshot,
        "metrics": {
            "candidate_count": int(len(candidates)),
            "current_vip_count": int(len(current_vip)),
            "precision_at_k": float(bt["precision_at_k"]),
            "lift_at_k": float(bt["lift_at_k"]) if np.isfinite(bt["lift_at_k"]) else 0.0,
            "roi_total": float(roi["ev_total"]),
            "roi_per_head": float(roi["ev_per_head"]),
            "response_rate": float(roi["p"]),
            "k_eval": int(k_eval),
        },
        "candidates": records(candidates, columns=columns, limit=limit),
        "current_vip": records(current_vip, columns=vip_columns, limit=limit),
        "candidate_df": candidates,
        "current_vip_df": current_vip,
    }


def percentile(series: pd.Series, value: Any) -> float | None:
    s = _numeric(series).dropna()
    if s.empty or pd.isna(value):
        return None
    return float((s <= float(value)).sum()) / float(len(s)) * 100


def describe_problem_action(feature: str, zval: float) -> tuple[str, str]:
    if feature == "CSFrequency":
        return (
            "상담 요청이 정상 고객보다 자주 발생합니다." if zval > 0 else "상담 이력은 많지 않지만 최근 주문 이력 확인이 필요합니다.",
            "시니어 상담을 배정해 최근 이슈를 정리하고 불만 원인을 해소합니다.",
        )
    if feature == "NegativeExperienceIndex":
        return (
            "불만/클레임 관련 신호가 정상 고객보다 많습니다." if zval > 0 else "부정 경험 지수는 낮지만 개별 이력 확인은 필요합니다.",
            "주요 클레임 유형을 정리하고 관련 티켓을 우선 처리합니다.",
        )
    if feature == "AverageSatisfactionScore":
        return (
            "만족도가 정상 고객보다 낮습니다." if zval < 0 else "만족도는 높은 편입니다.",
            "케어 콜과 보상 제안으로 불만 요소를 확인합니다." if zval < 0 else "긍정 후기를 유도하고 충성 고객 프로그램을 제안합니다.",
        )
    if feature in {"RecencyProxy", "AvgPurchaseInterval", "PurchaseFrequency"}:
        return (
            "최근 이용·구매 빈도가 줄어든 상태입니다.",
            "재방문 쿠폰과 재구매 리마인드로 다음 구매 시점을 앞당깁니다.",
        )
    if feature in {"EmailEngagementRate", "TotalEngagementScore"}:
        return (
            "앱/이메일 활동 수준이 낮아진 상태입니다.",
            "앱 푸시나 SMS로 채널을 전환하고 재온보딩 메시지를 발송합니다.",
        )
    return "정상 고객과 다른 패턴을 보입니다.", "상세 이력을 보고 원인을 파악한 뒤 맞춤 케어를 진행합니다."


def severity_label(zval: float) -> str:
    sev = abs(float(zval))
    if sev >= 2.5:
        return "매우 큼"
    if sev >= 1.5:
        return "큼"
    if sev >= 1.0:
        return "보통"
    return "작음"


def customer_drivers(df: pd.DataFrame, row: pd.Series) -> list[dict[str, Any]]:
    driver_cols = [c for c in RISK_DRIVER_NAMES if c in df.columns and c in row.index]
    if not driver_cols:
        return []

    if "Both_ChurnFlag" in df.columns:
        healthy = df[pd.to_numeric(df["Both_ChurnFlag"], errors="coerce").fillna(0).astype(int) == 0]
    else:
        healthy = df
    healthy = healthy[driver_cols].apply(pd.to_numeric, errors="coerce")
    mu = healthy.mean(numeric_only=True)
    sigma = healthy.std(numeric_only=True).replace(0, 1e-6).fillna(1e-6)

    z = ((_numeric(row[driver_cols]) - mu) / sigma).astype(float)
    z = z.reindex(z.abs().sort_values(ascending=False).index)

    rows: list[dict[str, Any]] = []
    for feature, zval in z.head(5).items():
        problem, action = describe_problem_action(feature, float(zval))
        rows.append(
            {
                "feature": feature,
                "label": RISK_DRIVER_NAMES.get(feature, feature),
                "current": clean_value(row[feature]),
                "healthy_avg": clean_value(mu[feature]),
                "z": float(zval),
                "severity": severity_label(float(zval)),
                "problem": problem,
                "action": action,
            }
        )
    return rows


def customer_profile(df: pd.DataFrame, actions: pd.DataFrame, customer_id: str) -> dict[str, Any] | None:
    match = df[df["CustomerID_clean"].astype(str) == str(customer_id)]
    if match.empty:
        return None
    row = match.iloc[0]

    both = int(row.get("Both_ChurnFlag", 0)) if "Both_ChurnFlag" in df.columns else 0
    if_flag = int(row.get("IF_ChurnFlag", 0)) if "IF_ChurnFlag" in df.columns else 0
    ae_flag = int(row.get("AE_ChurnFlag", 0)) if "AE_ChurnFlag" in df.columns else 0
    if both:
        status = {"badge": "고위험", "message": "즉시 관리 필요", "signals": "이상 행동 + 패턴 변화"}
    elif if_flag:
        status = {"badge": "주의", "message": "불만/이상 행동 신호", "signals": "이상 행동"}
    elif ae_flag:
        status = {"badge": "관찰", "message": "이용 패턴 감소 신호", "signals": "패턴 변화"}
    else:
        status = {"badge": "정상", "message": "특이 신호 없음", "signals": ""}

    basic_cols = ["CustomerID_clean", "GenderLabel", "Age", "IncomeLevel", "CustomerTenure", "RepeatCustomer", "RepeatAndPremiumFlag", "CustomerType"]
    activity_cols = [
        "TotalPurchases",
        "AverageOrderValue",
        "CustomerLifetimeValue",
        "PurchaseFrequency",
        "AvgPurchaseInterval",
        "CSFrequency",
        "AverageSatisfactionScore",
        "NegativeExperienceIndex",
        "EmailEngagementRate",
        "TotalEngagementScore",
        "RecencyProxy",
    ]
    risk_cols = ["RiskScore100", "RiskLevel", "ChurnRiskScore", "IF_AnomalyScore", "AE_ReconError"]

    feature_positions = []
    for col in [c for c in activity_cols if c in df.columns]:
        value = row[col]
        pct = percentile(df[col], value)
        direction = RISK_DRIVER_DIR.get(col, "neutral")
        if pct is None:
            risk_pct = None
        elif direction == "higher_worse":
            risk_pct = pct
        elif direction == "lower_worse":
            risk_pct = 100 - pct
        else:
            risk_pct = abs(pct - 50) * 2
        feature_positions.append(
            {
                "label": label_for(col),
                "value": clean_value(value),
                "percentile": pct,
                "risk_percent": risk_pct,
                "direction": {"higher_worse": "높을수록 위험", "lower_worse": "낮을수록 위험"}.get(direction, "중립"),
            }
        )

    history = actions[actions["customer_id"].astype(str) == str(customer_id)].copy() if not actions.empty else pd.DataFrame()
    history_rows = records(history, columns=["ts", "action", "owner", "status", "note"], limit=50)

    drivers = customer_drivers(df, row)
    default_sms = generate_sms(customer_id, drivers=drivers)

    return {
        "row": {k: clean_value(v) for k, v in row.to_dict().items()},
        "status": status,
        "basic": [(label_for(c), clean_value(row[c])) for c in basic_cols if c in row.index],
        "activity": [(label_for(c), clean_value(row[c])) for c in activity_cols if c in row.index],
        "risk": [(label_for(c), clean_value(row[c])) for c in risk_cols if c in row.index],
        "feature_positions": feature_positions,
        "drivers": drivers,
        "history": history_rows,
        "default_sms": default_sms,
    }


def sms_segments_korean(text: str) -> dict[str, int]:
    length = len(text or "")
    if length <= 70:
        return {"segments": 1, "remaining": 70 - length, "length": length}
    segments = 1 + math.ceil((length - 70) / 67.0)
    remaining = (67 - ((length - 70) % 67)) % 67
    return {"segments": segments, "remaining": remaining, "length": length}


def _target_limit(target_segments: int) -> int:
    return 70 if target_segments <= 1 else 70 + 67 * (target_segments - 1)


def _driver_reason(drivers: list[dict[str, Any]] | None) -> str | None:
    if not drivers:
        return None
    feature = drivers[0]["feature"]
    if feature == "CSFrequency":
        return "최근 상담이 자주 발생해 많이 번거로우셨을 수 있습니다."
    if feature == "NegativeExperienceIndex":
        return "이용 과정에서 불편이나 클레임이 있었던 것으로 보입니다."
    if feature == "AverageSatisfactionScore":
        return "만족도 응답에서 기대에 못 미친 부분이 있었습니다."
    if feature in {"RecencyProxy", "AvgPurchaseInterval", "PurchaseFrequency"}:
        return "최근 이용·구매 빈도가 줄어든 상태입니다."
    if feature in {"EmailEngagementRate", "TotalEngagementScore"}:
        return "앱·이메일 활동이 예전보다 줄어든 상태입니다."
    return "이용 패턴에 변동이 있는 고객으로 분석되었습니다."


def _detect_theme(drivers: list[dict[str, Any]] | None) -> str:
    if not drivers:
        return "promo"
    first = drivers[0]["feature"]
    z = float(drivers[0]["z"])
    if first in {"CSFrequency", "NegativeExperienceIndex"} and z > 0:
        return "care"
    if first == "AverageSatisfactionScore" and z < 0:
        return "care"
    if first in {"RecencyProxy", "AvgPurchaseInterval"} or (first == "PurchaseFrequency" and z < 0):
        return "winback"
    if first in {"EmailEngagementRate", "TotalEngagementScore"}:
        return "engage"
    return "promo"


def _fit_to_target(text: str, target_segments: int) -> str:
    limit = _target_limit(target_segments)
    if len(text) <= limit:
        return text
    trimmed = text
    for marker in [" 수신거부", " 문의:", " 바로가기:"]:
        idx = trimmed.rfind(marker)
        if idx != -1 and len(trimmed) > limit:
            trimmed = trimmed[:idx].strip()
    if len(trimmed) <= limit:
        return trimmed
    trimmed = trimmed[:limit]
    last_space = trimmed.rfind(" ")
    if last_space > limit * 0.5:
        trimmed = trimmed[:last_space]
    return trimmed.rstrip(" ,.;") + "…"


def generate_sms(
    customer_id: str,
    brand: str = "브랜드",
    benefit: str = "5% 할인 쿠폰",
    expiry: str | dt.date | None = None,
    tone: str = "정중",
    theme: str = "auto",
    landing_url: str = "",
    cs_contact: str = "",
    optout: str = "수신거부: 수신중지",
    target_segments: int = 1,
    drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if expiry is None:
        expiry = dt.date.today() + dt.timedelta(days=7)
    exp_str = expiry.strftime("%Y-%m-%d") if isinstance(expiry, dt.date) else str(expiry or "")
    selected_theme = _detect_theme(drivers) if theme == "auto" else theme
    reason = _driver_reason(drivers) if target_segments > 1 else None

    if tone == "친근":
        hi_short = f"[{brand}] {customer_id}님"
        hi_long = f"[{brand}] {customer_id}님 안녕하세요."
    elif tone == "긴급":
        hi_short = f"[{brand}] {customer_id} 고객님"
        hi_long = f"[{brand}] {customer_id} 고객님, 중요한 안내드립니다."
    else:
        hi_short = f"[{brand}] {customer_id} 고객님"
        hi_long = f"[{brand}] {customer_id} 고객님 안녕하세요."

    reason_text = f"{reason} " if reason else ""
    if target_segments <= 1:
        variants = {
            "care": [
                f"{hi_short}, 이용 중 불편을 드려 죄송합니다. 사과의 마음으로 {benefit}을 드립니다.",
                f"{hi_short}, 서비스 이용에 불편이 있으셨다면 죄송합니다. {benefit}을 준비했습니다.",
            ],
            "winback": [
                f"{hi_short}, 오랜만에 인사드립니다. 다시 방문 시 {benefit}을 드립니다.",
                f"{hi_short}, 최근 이용이 줄어 아쉬운 마음에 {benefit}을 준비했습니다.",
            ],
            "engage": [
                f"{hi_short}, 새 혜택과 이벤트가 열렸습니다. {benefit}을 확인해 주세요.",
                f"{hi_short}, 혜택을 놓치지 않도록 {benefit}을 안내드립니다.",
            ],
            "promo": [
                f"{hi_short}께 {benefit} 혜택을 준비했습니다.",
                f"{hi_short}, 지금 {benefit}을 이용하실 수 있습니다.",
            ],
        }
    else:
        variants = {
            "care": [
                f"{hi_long} 이용 중 불편을 드려 죄송합니다. {reason_text}사과의 마음으로 {benefit}을 준비했으며 {exp_str}까지 사용 가능합니다.",
                f"{hi_long} 만족스럽지 못하셨던 부분이 있었던 것 같습니다. {reason_text}{benefit}을 제공해 드립니다.",
            ],
            "winback": [
                f"{hi_long} 요즘 자주 뵙지 못해 먼저 연락드립니다. {reason_text}다시 방문 시 {benefit}을 드립니다. {exp_str}까지 사용 가능합니다.",
                f"{hi_long} 한동안 이용이 뜸하셔서 아쉬운 마음에 연락드립니다. {reason_text}{benefit}으로 다시 혜택을 경험해 보세요.",
            ],
            "engage": [
                f"{hi_long} 새 혜택과 이벤트가 업데이트되었습니다. {reason_text}고객님께 맞는 {benefit}을 {exp_str} 전에 확인해 주세요.",
                f"{hi_long} 혜택과 알림을 더 알차게 이용하실 수 있도록 {benefit}을 추가했습니다. {reason_text}{exp_str}까지입니다.",
            ],
            "promo": [
                f"{hi_long} 고객님께 어울리는 {benefit}을 준비했습니다. {exp_str}까지 사용 가능하니 쇼핑에 참고 부탁드립니다.",
                f"{hi_long} 현재 고객님께 제공되는 {benefit}이 오픈되었습니다. {exp_str} 전까지 자유롭게 사용해 보세요.",
            ],
        }

    url = f" 바로가기: {landing_url}" if landing_url else ""
    cs = f" 문의: {cs_contact}" if cs_contact else ""
    oo = f" {optout}" if optout else ""
    candidates = [(text + url + cs + oo).strip() for text in variants.get(selected_theme, variants["promo"])]
    fitted = [_fit_to_target(text, int(target_segments)) for text in candidates]
    message = min(fitted, key=len)
    return {
        "theme": selected_theme,
        "message": message,
        "alternatives": fitted,
        "segments": sms_segments_korean(message),
    }
