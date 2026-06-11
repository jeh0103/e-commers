# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MAIN_CSV = ROOT_DIR / "ecommerce_customer_churn_hybrid_with_id.csv"
FEATURED_CSV = ROOT_DIR / "ecommerce_customer_data_featured.csv"
ACTIONS_DB = ROOT_DIR / "actions.db"

ACTIONS_LOOKBACK_DAYS = 7
ACTIONS_BENEFIT_KEYWORDS = ("쿠폰", "혜택", "VIP")

DEFAULT_CODE_TO_LABEL_KO = {
    0: "여성",
    1: "여성",
    2: "남성",
    3: "남성",
    4: "기타/미상",
    5: "응답거부",
    6: "기타/미상",
}

COL_LABEL_KO = {
    "CustomerID": "고객ID",
    "CustomerID_clean": "고객ID",
    "GenderLabel": "성별",
    "Age": "나이",
    "IncomeLevel": "소득 수준",
    "CustomerTenure": "이용 개월",
    "RepeatCustomer": "재구매 고객",
    "RepeatAndPremiumFlag": "우수고객 여부",
    "TotalPurchases": "총 구매 횟수",
    "AverageOrderValue": "평균 주문 금액",
    "CustomerLifetimeValue": "고객 가치",
    "PurchaseFrequency": "구매 빈도",
    "AvgPurchaseInterval": "평균 구매 간격",
    "CSFrequency": "상담 빈도",
    "AverageSatisfactionScore": "평균 만족도",
    "NegativeExperienceIndex": "부정 경험 지수",
    "EmailEngagementRate": "이메일 참여율",
    "MobileAppUsage": "모바일앱 사용",
    "TotalEngagementScore": "총 참여 점수",
    "RecencyProxy": "휴면 징후 지수",
    "ChurnRiskScore": "이탈 위험 원점수",
    "RiskScore100": "이탈 위험 점수",
    "RiskLevel": "위험 수준",
    "IF_AnomalyScore": "이상 행동 점수",
    "AE_ReconError": "패턴 변화 점수",
    "BehaviorClusterName": "고객 유형",
    "CustomerType": "고객 유형",
    "VIP잠재지수": "VIP 점수",
    "coverage": "데이터 충분도",
}


CUSTOMER_TYPE_LABEL_KO = {
    "고활성·불만多(고위험)": "집중 관리 고객",
    "저활성·불만多(고위험)": "이탈 위험 고객",
    "소액 저빈도(일반층)": "일반 고객",
    "대규모 안정 고객(저위험)": "핵심 고객",
    "고가·고가치 고객(VIP형)": "VIP 전환 후보",
}


def label_for(column: str) -> str:
    return COL_LABEL_KO.get(column, column)


def _generated_ids(length: int, index: pd.Index | None = None) -> pd.Series:
    idx = index if index is not None else pd.RangeIndex(length)
    return pd.Series(np.arange(1, length + 1), index=idx).map(lambda i: f"CUST{i:05d}")


def _clean_id_value(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "nat", "null"}:
        return np.nan
    return text


def ensure_customer_id_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "CustomerID_clean" in out.columns:
        clean = out["CustomerID_clean"].map(_clean_id_value)
    elif "CustomerID" in out.columns:
        clean = out["CustomerID"].map(_clean_id_value)
    else:
        clean = _generated_ids(len(out), out.index)

    bad = clean.isna() | clean.astype(str).str.strip().eq("")
    if bad.any():
        fallback = _generated_ids(len(out), out.index)
        clean.loc[bad] = fallback.loc[bad]

    out["CustomerID_clean"] = clean.astype(str)
    if "CustomerID" not in out.columns:
        out["CustomerID"] = out["CustomerID_clean"]
    return out


def normalize_gender(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None

    low = text.lower()
    if low in {"m", "male", "man", "남", "남성"}:
        return "남성"
    if low in {"f", "female", "woman", "여", "여성"}:
        return "여성"
    if low in {"prefer not to say", "decline to state", "no answer", "응답거부"}:
        return "응답거부"
    if low in {"non-binary", "nonbinary", "genderqueer", "agender", "nb", "other", "기타"}:
        return "기타"

    try:
        number = float(text)
        if number.is_integer():
            return DEFAULT_CODE_TO_LABEL_KO.get(int(number))
    except Exception:
        pass
    return text if any("\uac00" <= ch <= "\ud7a3" for ch in text) else "기타"


def ensure_gender_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    label = pd.Series([None] * len(out), index=out.index, dtype="object")

    if "GenderLabel" in out.columns:
        label = out["GenderLabel"].map(normalize_gender)
    if "Gender" in out.columns:
        from_gender = out["Gender"].map(normalize_gender)
        label = label.fillna(from_gender)

    out["GenderLabel"] = label.fillna("미상")
    return out


def clean_customer_type(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "미분류"
    text = str(value).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        if len(left.strip()) <= 3:
            text = right.strip()
    if not text:
        return "미분류"
    return CUSTOMER_TYPE_LABEL_KO.get(text, text.replace("多", " 많음"))


def compute_risk_score_100(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        return pd.Series(np.nan, index=series.index)
    return (numeric.rank(pct=True) * 100.0).round(0)


def risk_level(score: Any) -> str:
    if pd.isna(score):
        return "정보없음"
    value = float(score)
    if value >= 90:
        return "매우 높음"
    if value >= 70:
        return "높음"
    if value >= 40:
        return "보통"
    if value >= 20:
        return "낮음"
    return "매우 낮음"


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_customer_id_clean(df)
    out = ensure_gender_label(out)

    cluster_col = "BehaviorClusterName" if "BehaviorClusterName" in out.columns else "BehaviorCluster"
    if cluster_col in out.columns:
        out["CustomerType"] = out[cluster_col].map(clean_customer_type)
    else:
        out["CustomerType"] = "미분류"

    if "ChurnRiskScore" in out.columns:
        out["RiskScore100"] = compute_risk_score_100(out["ChurnRiskScore"])
        out["RiskLevel"] = out["RiskScore100"].map(risk_level)
    else:
        out["RiskScore100"] = np.nan
        out["RiskLevel"] = "정보없음"

    return out


@lru_cache(maxsize=1)
def load_customers_cached() -> pd.DataFrame:
    if not MAIN_CSV.exists():
        raise FileNotFoundError(f"{MAIN_CSV.name} 파일을 찾을 수 없습니다.")
    return _add_derived_columns(pd.read_csv(MAIN_CSV))


def load_customers() -> pd.DataFrame:
    return load_customers_cached().copy()


@lru_cache(maxsize=1)
def load_featured_cached() -> pd.DataFrame:
    if not FEATURED_CSV.exists():
        return pd.DataFrame()
    return _add_derived_columns(pd.read_csv(FEATURED_CSV))


def load_featured() -> pd.DataFrame:
    return load_featured_cached().copy()


def clear_data_cache() -> None:
    load_customers_cached.cache_clear()
    load_featured_cached.cache_clear()


def actions_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(ACTIONS_DB, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            action TEXT,
            note TEXT,
            ts TEXT,
            owner TEXT,
            status TEXT
        )
        """
    )
    conn.commit()
    return conn


def load_actions() -> pd.DataFrame:
    if not ACTIONS_DB.exists():
        return pd.DataFrame(columns=["id", "customer_id", "action", "note", "ts", "owner", "status"])

    conn = actions_connection()
    try:
        df = pd.read_sql_query(
            "SELECT id, customer_id, action, note, ts, owner, status FROM actions ORDER BY ts DESC",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def insert_action(customer_id: str, action: str, note: str = "", owner: str = "", status: str = "open") -> int:
    now = dt.datetime.utcnow().isoformat()
    conn = actions_connection()
    try:
        cur = conn.execute(
            "INSERT INTO actions (customer_id, action, note, ts, owner, status) VALUES (?, ?, ?, ?, ?, ?)",
            (customer_id, action, note, now, owner, status),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def recent_action_sets(actions: pd.DataFrame, lookback_days: int = ACTIONS_LOOKBACK_DAYS) -> tuple[set[str], set[str]]:
    if actions.empty or "ts" not in actions.columns:
        return set(), set()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    recent = actions[actions["ts"] >= cutoff].copy()
    contacted = set(recent["customer_id"].dropna().astype(str))
    if recent.empty:
        return contacted, set()

    pattern = "|".join(ACTIONS_BENEFIT_KEYWORDS)
    benefit_mask = recent["action"].fillna("").astype(str).str.contains(pattern, case=False, na=False)
    benefit = set(recent.loc[benefit_mask, "customer_id"].dropna().astype(str))
    return contacted, benefit


def apply_filters(
    df: pd.DataFrame,
    age_min: float | None = None,
    age_max: float | None = None,
    genders: Iterable[str] | None = None,
    premium: str | None = None,
    use_dynamic: bool = False,
    if_threshold: float | None = None,
    ae_threshold: float | None = None,
) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()

    if "Age" in out.columns:
        if age_min is not None:
            out = out[pd.to_numeric(out["Age"], errors="coerce") >= float(age_min)]
        if age_max is not None:
            out = out[pd.to_numeric(out["Age"], errors="coerce") <= float(age_max)]

    gender_values = [g for g in (genders or []) if g]
    if gender_values and "GenderLabel" in out.columns:
        out = out[out["GenderLabel"].isin(gender_values)]

    if premium and premium != "all" and "RepeatAndPremiumFlag" in out.columns:
        expected = 1 if premium == "yes" else 0
        out = out[pd.to_numeric(out["RepeatAndPremiumFlag"], errors="coerce").fillna(-1).astype(int) == expected]

    flag_col: str | None = "Both_ChurnFlag" if "Both_ChurnFlag" in out.columns else None
    if use_dynamic and {"IF_AnomalyScore", "AE_ReconError"}.issubset(out.columns):
        out = out.copy()
        if if_threshold is None:
            if_threshold = float(pd.to_numeric(out["IF_AnomalyScore"], errors="coerce").quantile(0.95))
        if ae_threshold is None:
            ae_threshold = float(pd.to_numeric(out["AE_ReconError"], errors="coerce").quantile(0.95))
        out["IF_ChurnFlag_dyn"] = (pd.to_numeric(out["IF_AnomalyScore"], errors="coerce") >= float(if_threshold)).astype(int)
        out["AE_ChurnFlag_dyn"] = (pd.to_numeric(out["AE_ReconError"], errors="coerce") >= float(ae_threshold)).astype(int)
        out["Both_ChurnFlag_dyn"] = (out["IF_ChurnFlag_dyn"] & out["AE_ChurnFlag_dyn"]).astype(int)
        flag_col = "Both_ChurnFlag_dyn"

    return out, flag_col


def filter_options(df: pd.DataFrame) -> dict[str, Any]:
    ages = pd.to_numeric(df.get("Age", pd.Series(dtype=float)), errors="coerce").dropna()
    if ages.empty:
        age_min, age_max = None, None
    else:
        age_min, age_max = int(np.floor(ages.min())), int(np.ceil(ages.max()))

    genders = []
    if "GenderLabel" in df.columns:
        genders = sorted(df["GenderLabel"].dropna().astype(str).unique().tolist())

    if_q95 = None
    ae_q95 = None
    if "IF_AnomalyScore" in df.columns:
        if_q95 = float(pd.to_numeric(df["IF_AnomalyScore"], errors="coerce").quantile(0.95))
    if "AE_ReconError" in df.columns:
        ae_q95 = float(pd.to_numeric(df["AE_ReconError"], errors="coerce").quantile(0.95))

    return {
        "age_min": age_min,
        "age_max": age_max,
        "genders": genders,
        "if_q95": if_q95,
        "ae_q95": ae_q95,
    }


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def records(df: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    view = df.copy()
    if columns is not None:
        view = view[[c for c in columns if c in view.columns]]
    if limit is not None:
        view = view.head(limit)
    return [{k: clean_value(v) for k, v in row.items()} for row in view.to_dict(orient="records")]


def format_kr_time(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Seoul").tz_localize(None)
        else:
            ts = ts + pd.Timedelta(hours=9)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)
