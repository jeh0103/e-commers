# utils_vip.py
# -*- coding: utf-8 -*-
"""
VIP 후보선정/백테스트/ROI 공통 유틸 모듈

운영 원칙 반영:
- CustomerID NaN: 리스트/CSV/링크에서 제외, 통계 포함 여부 옵션
- 핵심지표 NaN: 있는 지표만 정규화+가중합 (coverage=사용지표수/가능지표수)
- 강한 단일 신호 허용: 상위 p%이면 coverage 미달이어도 후보 인정
- 최종점수: raw × (0.5 + 0.5 × sqrt(coverage))
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# 점수에 기여하는 기본 피처
POS_COLS: List[str] = [
    "PurchaseFrequency", "AverageOrderValue", "TotalPurchases",
    "TotalEngagementScore", "EmailEngagementRate", "MobileAppUsage",
]
# 값이 작을수록 좋은 피처(정규화 시 반전)
NEG_COLS: List[str] = [
    "AvgPurchaseInterval", "NegativeExperienceIndex", "CSFrequency",
]

def _qnorm01(ref: pd.Series, val: pd.Series, lo: float = 0.05, hi: float = 0.95, invert: bool = False) -> pd.Series:
    """분위 기반 0~1 정규화(이상치 완화)."""
    r = pd.to_numeric(ref, errors="coerce").dropna()
    v = pd.to_numeric(val, errors="coerce")
    if r.empty:
        rmin, rmax = v.min(skipna=True), v.max(skipna=True)
    else:
        rmin, rmax = float(r.quantile(lo)), float(r.quantile(hi))
        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
            rmin, rmax = float(r.min()), float(r.max())
    x = (v - rmin) / max(1e-9, (rmax - rmin))
    x = x.clip(0, 1)
    return (1 - x) if invert else x

def compute_vip_propensity_score(
    df: pd.DataFrame,
    ref_df: Optional[pd.DataFrame] = None,
    pos_cols: List[str] = POS_COLS,
    neg_cols: List[str] = NEG_COLS,
    weights: Optional[Dict[str, float]] = None,
    id_col: str = "CustomerID_clean",
) -> pd.DataFrame:
    """
    VIP 잠재지수 산출(원시/최종/coverage/사용피처 목록 포함).
    - 있는 지표만 정규화→가중합 → raw(0~1)
    - coverage = 사용지표수 / 가능지표수
    - final = raw × (0.5 + 0.5 × sqrt(coverage)) → 0~1 → 0~100 변환
    """
    ref = ref_df if ref_df is not None else df
    cols = [c for c in (pos_cols + neg_cols) if c in df.columns]
    out_index = df.index

    # 아무 지표도 없으면 0 반환
    if not cols:
        out = pd.DataFrame(index=out_index)
        if id_col in df.columns:
            out[id_col] = df[id_col].values
        out["VIP잠재지수_raw"] = 0.0
        out["coverage"] = 0.0
        out["VIP잠재지수"] = 0.0
        out["score_used_cols"] = [[] for _ in range(len(out))]
        return out

    # 기본 가중치(합 1.0)
    if weights is None:
        weights = {
            "PurchaseFrequency": 0.22, "AverageOrderValue": 0.20,
            "TotalPurchases": 0.15, "TotalEngagementScore": 0.15,
            "EmailEngagementRate": 0.08, "MobileAppUsage": 0.08,
            "AvgPurchaseInterval": 0.06, "NegativeExperienceIndex": 0.04,
            "CSFrequency": 0.02,
        }
    w = {c: float(weights.get(c, 0.0)) for c in cols}

    # 정규화
    normed = {}
    for c in cols:
        if c in pos_cols:
            normed[c] = _qnorm01(ref[c], df[c], invert=False)
        else:
            normed[c] = _qnorm01(ref[c], df[c], invert=True)
    norm_df = pd.DataFrame(normed, index=out_index)

    # 행별 "있는 값"만 가중합
    valid = norm_df.notna()
    row_weights = pd.DataFrame({c: w[c] for c in norm_df.columns}, index=out_index)
    row_weights = row_weights.where(valid, 0.0)
    denom = row_weights.sum(axis=1).replace(0, np.nan)
    raw = (norm_df * row_weights).sum(axis=1) / denom
    raw = raw.fillna(0.0)

    # coverage 계산
    possible_cols = [c for c in cols if w.get(c, 0) > 0]
    used_count = valid[possible_cols].sum(axis=1)
    coverage = (used_count / max(1, len(possible_cols))).astype(float)

    # coverage 보정 및 0~100 변환
    final01 = raw * (0.5 + 0.5 * np.sqrt(coverage))
    final100 = (final01 * 100).clip(0, 100).round(0)

    out = pd.DataFrame(index=out_index)
    if id_col in df.columns:
        out[id_col] = df[id_col].values
    out["VIP잠재지수_raw"] = raw.round(4)
    out["coverage"] = coverage.round(4)
    out["VIP잠재지수"] = final100.astype(float)
    out["score_used_cols"] = [
        [c for c in possible_cols if pd.notna(norm_df.loc[i, c])] for i in out_index
    ]
    return out

def select_vip_candidates(
    df_scored: pd.DataFrame,
    id_col: str = "CustomerID_clean",
    mode: str = "threshold",             
    k: int = 100,
    thr: float = 80.0,
    coverage_min_n: int = 3,
    strong_signal_pct: float = 95.0,
    pos_cols: List[str] = POS_COLS,
    include_nan_id_in_stats: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    후보선정(운영 원칙):
      - coverage_n >= coverage_min_n  또는  강한 단일 신호(positive col 상위 p%)
      - mode='threshold' → VIP잠재지수 ≥ thr,  mode='topk' → VIP잠재지수 상위 k
      - 반환: (리스트 DF, 스냅샷 dict)
    """
    if "VIP잠재지수" not in df_scored.columns:
        raise ValueError("VIP잠재지수 컬럼이 없습니다. compute_vip_propensity_score() 먼저 호출하세요.")

    # 강한 단일 신호
    strong = pd.Series(False, index=df_scored.index)
    for c in pos_cols:
        if c in df_scored.columns:
            s = pd.to_numeric(df_scored[c], errors="coerce")
            cutoff = float(s.quantile(strong_signal_pct / 100.0))
            strong |= (s >= cutoff)

    # coverage_n
    cov_n = df_scored.get("score_used_cols", pd.Series([[]] * len(df_scored), index=df_scored.index)).apply(len)
    base_rule = (cov_n >= int(coverage_min_n)) | strong

    pool = df_scored[base_rule].copy()

    if mode == "topk":
        pool = pool.sort_values("VIP잠재지수", ascending=False)
        selected = pool.head(int(k)).copy()
        threshold_used = float(selected["VIP잠재지수"].min()) if len(selected) else float("nan")
    else:
        selected = pool[pool["VIP잠재지수"] >= float(thr)].copy()
        selected = selected.sort_values("VIP잠재지수", ascending=False)
        threshold_used = float(thr)

    # NaN ID는 리스트/CSV 제외
    list_df = selected.copy()
    if id_col in list_df.columns:
        list_df = list_df[list_df[id_col].notna()]

    in_stats_n = len(selected) if include_nan_id_in_stats else len(list_df)
    snap = {
        "count_selected": int(in_stats_n),
        "threshold_used": threshold_used,
        "mode": mode,
        "coverage_min_n": int(coverage_min_n),
        "strong_signal_pct": float(strong_signal_pct),
    }
    return list_df, snap

# ---------------- Backtest / ROI ---------------- #

def precision_at_k(y_true: pd.Series, scores: pd.Series, k: int) -> float:
    s = pd.DataFrame({
        "y": pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int),
        "s": pd.to_numeric(scores, errors="coerce").fillna(-np.inf),
    })
    s = s.sort_values("s", ascending=False).head(int(k))
    return float(s["y"].sum()) / len(s) if len(s) else 0.0

def lift_at_k(y_true: pd.Series, scores: pd.Series, k: int) -> float:
    p_at_k = precision_at_k(y_true, scores, k)
    base = float(pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).mean())
    return (p_at_k / base) if base > 0 else float("nan")

def make_proxy_label(df: pd.DataFrame) -> pd.Series:
    """
    프락시 라벨 우선순위:
    1) NextPurchaseDays ≤ 30 → 1
    2) RecencyProxy 하위 20% → 1
    3) AvgPurchaseInterval 하위 20% → 1
    없으면 전부 0
    """
    if "NextPurchaseDays" in df.columns:
        return (pd.to_numeric(df["NextPurchaseDays"], errors="coerce") <= 30).astype(int)
    if "RecencyProxy" in df.columns:
        r = pd.to_numeric(df["RecencyProxy"], errors="coerce")
        thr = float(r.quantile(0.20))
        return (r <= thr).astype(int)
    if "AvgPurchaseInterval" in df.columns:
        a = pd.to_numeric(df["AvgPurchaseInterval"], errors="coerce")
        thr = float(a.quantile(0.20))
        return (a <= thr).astype(int)
    return pd.Series(0, index=df.index, dtype=int)

def backtest_metrics(
    df_scored: pd.DataFrame,
    score_col: str = "VIP잠재지수",
    label_col: Optional[str] = None,
    k: int = 100,
) -> Dict[str, float]:
    y = df_scored[label_col] if (label_col and label_col in df_scored.columns) else make_proxy_label(df_scored)
    s = pd.to_numeric(df_scored[score_col], errors="coerce").fillna(-np.inf)
    Pk = precision_at_k(y, s, k)
    Lk = lift_at_k(y, s, k)
    base = float(pd.to_numeric(y, errors="coerce").mean())
    return {"precision_at_k": Pk, "lift_at_k": Lk, "base_rate": base}

def roi_for_k(
    df_scored: pd.DataFrame,
    k: int = 100,
    score_col: str = "VIP잠재지수",
    label_col: Optional[str] = None,
    avg_order_value: float = 50000,   # 평균 객단가(₩)
    gross_margin: float = 0.35,       # 매출 총이익률
    cost_per_contact: float = 1000,   # 1인당 캠페인 비용
    baseline_response: Optional[float] = None,
) -> Dict[str, float]:
    """
    간단 ROI 추정: EV = k × (p × AOV × margin − cost)
    - 라벨 있으면 top-k 실제 반응률
    - 없으면 프락시 반응률(보수적으로 0.8배)
    """
    s = pd.to_numeric(df_scored[score_col], errors="coerce").fillna(-np.inf)
    idx = s.sort_values(ascending=False).index[:int(k)]

    if label_col and (label_col in df_scored.columns):
        y = pd.to_numeric(df_scored[label_col], errors="coerce").fillna(0).astype(int)
        p = float(y.loc[idx].mean())
    else:
        if baseline_response is None:
            proxy = make_proxy_label(df_scored)
            p = float(proxy.loc[idx].mean()) * 0.8  # 보수적
        else:
            p = float(baseline_response)

    unit_margin = avg_order_value * gross_margin
    ev_per_head = (p * unit_margin) - cost_per_contact
    ev_total = ev_per_head * int(k)
    return {"p": p, "unit_margin": unit_margin, "ev_per_head": ev_per_head, "ev_total": ev_total}
