# pages/04_Customer_Type_List.py
# -*- coding: utf-8 -*-
import os
import sqlite3
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="고객유형 고객 목록", layout="wide")

DETAIL_PAGE_SLUG = "Customer_Detail"  # pages/01_Customer_Detail.py → /Customer_Detail
ACTIONS_LOOKBACK_DAYS = 7
ACTIONS_BENEFIT_KEYWORDS = ["쿠폰", "혜택", "VIP"]


# -------------------------------
# Query-param helpers (new/old Streamlit 호환)
# -------------------------------
def qp_get(name: str):
    try:
        v = st.query_params.get(name)
    except Exception:
        v = st.experimental_get_query_params().get(name)
    if isinstance(v, list):
        v = v[0] if v else None
    return v


def qp_set(**kwargs):
    try:
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)


# -------------------------------
# Gender label helpers (대시보드와 동일한 톤)
# -------------------------------
DEFAULT_CODE_TO_LABEL_KO = {
    1: "여성",
    3: "남성",
    5: "응답거부",
    4: "기타/미상",
    2: "남성",
    0: "여성",
}


def ensure_gender_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 이미 문자열 라벨이면 그대로(결측만 보완)
    if "GenderLabel" in out.columns and not pd.api.types.is_numeric_dtype(out["GenderLabel"]):
        out["GenderLabel"] = out["GenderLabel"].fillna("미상").astype(str).replace({"nan": "미상"})
        return out

    # GenderLabel이 숫자라면(0~3) → 남/여로 정규화
    if "GenderLabel" in out.columns and pd.api.types.is_numeric_dtype(out["GenderLabel"]):
        code_map = {0: "여성", 1: "여성", 2: "남성", 3: "남성"}
        out["GenderLabel"] = out["GenderLabel"].map(code_map).fillna("미상")
        return out

    # Gender 코드로 보완
    if "Gender" in out.columns:
        out["GenderLabel"] = out["Gender"].map(DEFAULT_CODE_TO_LABEL_KO).fillna("미상")
    else:
        out["GenderLabel"] = "미상"
    return out


def ensure_customer_id_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "CustomerID_clean" in out.columns:
        # 결측만 보완
        mask_bad = out["CustomerID_clean"].isna() | out["CustomerID_clean"].astype(str).str.strip().eq("")
        if mask_bad.any():
            fallback = pd.Series(np.arange(1, len(out) + 1), index=out.index).map(lambda i: f"CUST{i:05d}")
            out.loc[mask_bad, "CustomerID_clean"] = fallback.loc[mask_bad]
        return out

    if "CustomerID" in out.columns:
        def _clean(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            return np.nan if (s == "" or s.lower() in {"nan", "none", "nat", "null"}) else s
        out["CustomerID_clean"] = out["CustomerID"].map(_clean)
    else:
        out["CustomerID_clean"] = pd.Series(np.arange(1, len(out) + 1)).map(lambda i: f"CUST{i:05d}")

    mask_bad = out["CustomerID_clean"].isna() | out["CustomerID_clean"].astype(str).str.strip().eq("")
    if mask_bad.any():
        fallback = pd.Series(np.arange(1, len(out) + 1), index=out.index).map(lambda i: f"CUST{i:05d}")
        out.loc[mask_bad, "CustomerID_clean"] = fallback.loc[mask_bad]

    return out


def clean_customer_type(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "미분류"
    s = str(x).strip()
    if ":" in s:
        left, right = s.split(":", 1)
        if len(left.strip()) <= 3:
            return right.strip()
    return s


def compute_risk_score_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    try:
        p99 = float(s.dropna().quantile(0.99))
        if not np.isfinite(p99) or p99 <= 0:
            p99 = 1.0
    except Exception:
        p99 = 1.0
    return (s / p99 * 100.0).clip(0, 100)


def risk_level(score100: float) -> str:
    try:
        v = float(score100)
    except Exception:
        return "미상"
    if v >= 80:
        return "매우 높음"
    if v >= 60:
        return "높음"
    if v >= 40:
        return "보통"
    if v >= 20:
        return "낮음"
    return "매우 낮음"


@st.cache_data(show_spinner=False)
def load_main():
    try:
        df = pd.read_csv("ecommerce_customer_churn_hybrid_with_id.csv")
    except Exception:
        return None
    df = ensure_customer_id_clean(df)
    df = ensure_gender_label(df)
    return df


@st.cache_data(show_spinner=False)
def load_actions():
    if not os.path.exists("actions.db"):
        return pd.DataFrame(columns=["customer_id", "action", "ts"])
    try:
        conn = sqlite3.connect("actions.db")
        adf = pd.read_sql_query("SELECT customer_id, action, ts FROM actions", conn)
        conn.close()
    except Exception:
        return pd.DataFrame(columns=["customer_id", "action", "ts"])

    adf["customer_id"] = adf["customer_id"].astype(str).str.strip()
    adf["action"] = adf["action"].astype(str)
    adf["ts"] = pd.to_datetime(adf["ts"], errors="coerce")
    return adf


# -------------------------------
# Data load
# -------------------------------
df = load_main()
if df is None or df.empty:
    st.error("메인 데이터(ecommerce_customer_churn_hybrid_with_id.csv)를 불러오지 못했습니다. 파일 경로/이름을 확인하세요.")
    st.stop()

actions_df = load_actions()

# 고객유형 컬럼 준비
cluster_col = "BehaviorClusterName" if "BehaviorClusterName" in df.columns else ("BehaviorCluster" if "BehaviorCluster" in df.columns else None)
if not cluster_col:
    st.error("고객유형(클러스터) 컬럼이 없습니다. BehaviorClusterName 또는 BehaviorCluster가 필요합니다.")
    st.stop()

df["고객유형"] = df[cluster_col].map(clean_customer_type)

# -------------------------------
# Header
# -------------------------------
try:
    st.page_link("app_enhanced.py", label="← 대시보드로", icon="🏠")
except Exception:
    # 구버전 호환: 링크 버튼이 없으면 텍스트 링크
    st.markdown("[← 대시보드로](/)")

st.title("🧩 고객유형 고객 목록")

all_types = sorted(pd.Series(df["고객유형"].dropna().unique()).tolist())
pref = qp_get("customer_type") or st.session_state.get("selected_customer_type")
default_idx = 0
if pref in all_types:
    default_idx = all_types.index(pref)

sel_type = st.selectbox("고객유형 선택", all_types, index=default_idx)
st.session_state["selected_customer_type"] = sel_type
qp_set(customer_type=sel_type)

st.caption(
    "이 표는 **해당 고객유형 내부에서 이탈 위험이 높은 순**으로 정렬됩니다. "
    "관리자는 여기서 '누구를 먼저 연락/혜택 대상으로 볼지'를 빠르게 정할 수 있어야 합니다."
)

# -------------------------------
# 최근 액션(연락/혜택) 집계
# -------------------------------
contacted_ids, benefit_ids = set(), set()
if not actions_df.empty:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=ACTIONS_LOOKBACK_DAYS)
    recent = actions_df[actions_df["ts"] >= cutoff].copy()

    contacted_ids = set(recent["customer_id"].dropna().astype(str))
    benefit_mask = recent["action"].fillna("").str.contains("|".join(ACTIONS_BENEFIT_KEYWORDS), case=False, na=False)
    benefit_ids = set(recent.loc[benefit_mask, "customer_id"].dropna().astype(str))

# -------------------------------
# Group slice + summary metrics
# -------------------------------
gdf = df[df["고객유형"] == sel_type].copy()
if "ChurnRiskScore" in gdf.columns:
    gdf["이탈 위험 점수(0~100)"] = compute_risk_score_100(gdf["ChurnRiskScore"])
else:
    gdf["이탈 위험 점수(0~100)"] = np.nan

# 고신뢰 이탈 플래그(가능하면 Both_* 우선)
flag_col = None
for cand in ["Both_ChurnFlag_dyn", "Both_ChurnFlag", "Both_ChurnFlagProxy"]:
    if cand in gdf.columns:
        flag_col = cand
        break

gdf["고신뢰 이탈"] = (gdf[flag_col] == 1) if flag_col else False

gdf["최근 7일 연락"] = gdf["CustomerID_clean"].astype(str).isin(contacted_ids)
gdf["최근 7일 혜택"] = gdf["CustomerID_clean"].astype(str).isin(benefit_ids)

c1, c2, c3, c4 = st.columns(4)
c1.metric("고객 수", f"{len(gdf):,}명")
c2.metric("고신뢰 이탈 비율", f"{(gdf['고신뢰 이탈'].mean() * 100.0 if len(gdf) else 0):.1f}%")
c3.metric("평균 이탈 위험(0~100)", "-" if gdf["이탈 위험 점수(0~100)"].isna().all() else f"{gdf['이탈 위험 점수(0~100)'].mean():.0f}")
c4.metric("최근 7일 연락 없음", f"{int((~gdf['최근 7일 연락']).sum()):,}명")

# -------------------------------
# 관리자 관점: 한눈에 보는 특징(전체 대비 차이)
# -------------------------------
key_cols = [
    ("PurchaseFrequency", "구매 빈도", "↓ 낮을수록 위험"),
    ("CSFrequency", "상담 빈도", "↑ 높을수록 위험"),
    ("AverageSatisfactionScore", "평균 만족도", "↓ 낮을수록 위험"),
    ("NegativeExperienceIndex", "부정 경험 지수", "↑ 높을수록 위험"),
    ("TotalEngagementScore", "총 참여 점수", "↓ 낮을수록 위험"),
]
rows = []
for col, label, direction in key_cols:
    if col not in df.columns:
        continue
    a = pd.to_numeric(df[col], errors="coerce")
    b = pd.to_numeric(gdf[col], errors="coerce")
    if b.dropna().empty or a.dropna().empty:
        continue
    a_mean = float(a.mean())
    b_mean = float(b.mean())
    if a_mean == 0:
        delta = b_mean - a_mean
        delta_txt = f"{delta:+.2f}"
    else:
        delta = (b_mean - a_mean) / abs(a_mean) * 100.0
        delta_txt = f"{delta:+.0f}%"
    rows.append((abs(delta), label, b_mean, a_mean, delta_txt, direction))

if rows:
    rows.sort(reverse=True)
    top = rows[:3]
    bullets = []
    for _, label, b_mean, a_mean, delta_txt, direction in top:
        bullets.append(f"- **{label}**: 유형 평균 {b_mean:.2f} (전체 대비 {delta_txt}) · {direction}")
    st.markdown("#### 👀 이 유형의 눈에 띄는 특징(전체 대비)")
    st.markdown("\n".join(bullets))
else:
    st.markdown("#### 👀 이 유형의 눈에 띄는 특징(전체 대비)")
    st.caption("비교할 수 있는 핵심 지표가 부족합니다.")


# -------------------------------
# 고객 목록
# -------------------------------
st.markdown("#### 📋 고객 리스트")

f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
with f1:
    q = st.text_input("고객ID 검색", value="", placeholder="예) CUST06884")
with f2:
    only_high = st.checkbox("고신뢰 이탈만", value=False)
with f3:
    only_no_contact = st.checkbox("최근 7일 연락 없음만", value=False)
with f4:
    min_risk = st.slider("최소 위험 점수", min_value=0, max_value=100, value=0, step=5)

view_df = gdf.copy()
if q.strip():
    view_df = view_df[view_df["CustomerID_clean"].astype(str).str.contains(q.strip(), case=False, na=False)]
if only_high:
    view_df = view_df[view_df["고신뢰 이탈"] == True]
if only_no_contact:
    view_df = view_df[view_df["최근 7일 연락"] == False]
if "이탈 위험 점수(0~100)" in view_df.columns:
    view_df = view_df[pd.to_numeric(view_df["이탈 위험 점수(0~100)"], errors="coerce").fillna(0) >= float(min_risk)]

# 표시 컬럼 구성(고객ID 최우선)
out = pd.DataFrame({
    "고객ID": view_df["CustomerID_clean"].astype(str),
    "성별": view_df.get("GenderLabel", "미상"),
    "나이": view_df.get("Age", np.nan),
    "리피트/프리미엄": view_df.get("RepeatAndPremiumFlag", np.nan),
    "최근 7일 연락": view_df["최근 7일 연락"].map({True: "✅", False: "—"}),
    "최근 7일 혜택": view_df["최근 7일 혜택"].map({True: "✅", False: "—"}),
    "위험수준": view_df["이탈 위험 점수(0~100)"].map(risk_level),
    "이탈 위험 점수(0~100)": pd.to_numeric(view_df["이탈 위험 점수(0~100)"], errors="coerce"),
    "고신뢰 이탈": view_df["고신뢰 이탈"].map({True: "예", False: "아니오"}),
})

# 리피트/프리미엄 보기 좋게
if "리피트/프리미엄" in out.columns:
    out["리피트/프리미엄"] = out["리피트/프리미엄"].map(lambda x: "예" if str(x) == "1" else "아니오")

out["상세"] = out["고객ID"].map(lambda cid: f"/{DETAIL_PAGE_SLUG}?customer_id={quote(str(cid))}")

# 정렬: 위험 점수 높은 순
if "이탈 위험 점수(0~100)" in out.columns:
    out = out.sort_values("이탈 위험 점수(0~100)", ascending=False, na_position="last")

st.dataframe(
    out,
    use_container_width=True,
    hide_index=True,
    column_config={
        "이탈 위험 점수(0~100)": st.column_config.NumberColumn(format="%.0f"),
        "상세": st.column_config.LinkColumn("상세", display_text="보기"),
    },
)

st.caption("표의 **상세-보기**를 누르면 고객 상세 화면으로 이동합니다.")
