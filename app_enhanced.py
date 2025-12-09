# app_enhanced.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os, json, sqlite3
from urllib.parse import quote, unquote

from utils_vip import compute_vip_propensity_score  # VIP 잠재지수 계산

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="고객 이탈 위험 대시보드", layout="wide")

# 상세 페이지 라우트 (pages/01_Customer_Detail.py → /Customer_Detail)
DETAIL_PAGE_SLUG = "Customer_Detail"  # 상세 링크에서 사용

# -------------------------------
# Query-param helpers
# -------------------------------
def qp_get(name: str):
    """Get query param for both new (st.query_params) and old (experimental_get_) APIs."""
    try:
        v = st.query_params.get(name)  # Streamlit >= 1.30+
    except Exception:
        v = st.experimental_get_query_params().get(name)  # older
    if isinstance(v, list):
        v = v[0] if v else None
    return v


def qp_set(**kwargs):
    """Set query params for both new and old APIs."""
    try:
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

# -------------------------------
# 화면 표시용 한글 라벨 맵(표시 전용; 내부 컬럼명은 그대로 사용)
# -------------------------------
KOR_COL = {
    "CustomerID_clean": "고객ID",
    "GenderLabel": "성별",
    "ChurnRiskScore": "이탈위험점수",
    "IF_AnomalyScore": "패턴이탈지수(IF)",
    "AE_ReconError": "정상패턴차이(AE)",
    "PurchaseFrequency": "구매 빈도(월 평균)",
    "CSFrequency": "상담 빈도(월 평균)",
    "AverageSatisfactionScore": "평균만족도",
    "NegativeExperienceIndex": "부정경험지수",
    "EmailEngagementRate": "이메일참여율",
    "TotalEngagementScore": "총참여점수",
    "AvgPurchaseInterval": "평균구매간격",
    "TotalPurchases": "총구매수",
    "AverageOrderValue": "평균주문금액",
    "CustomerLifetimeValue": "고객생애가치",
    "MobileAppUsage": "모바일앱사용",
    "CustomerServiceInteractions": "고객센터상담수",
    "Age": "나이",
    "RepeatAndPremiumFlag": "리피트/프리미엄",
    # VIP / 오늘 연락 대상용
    "VIP잠재지수": "VIP전환지수",
    "coverage": "데이터충분도",
    # 위험도 0~100 + 등급
    "RiskScore100": "이탈 위험 점수(0~100)",
    "RiskLevel": "위험 수준",
}

def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: KOR_COL.get(c, c) for c in df.columns})

# -------------------------------
# Gender standardization helpers
# -------------------------------
DEFAULT_CODE_TO_LABEL_KO = {
    1: "여성",
    3: "남성",
    5: "응답거부",
    4: "기타/미상",
    2: "남성",
    0: "여성",
}


def _normalize_gender_text_to_label_ko(x) -> str:
    """원본 문자열 성별을 한국어 라벨로 표준화."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "미상"
    s = str(x).strip().lower()
    if s in {"m", "male", "man", "남", "남성"}:
        return "남성"
    if s in {"f", "female", "woman", "여", "여성"}:
        return "여성"
    if s in {"prefer not to say", "decline to state", "no answer"}:
        return "응답거부"
    if s in {"non-binary", "nonbinary", "genderqueer", "agender", "nb"}:
        return "기타"
    if s in {"other", "기타"}:
        return "기타"
    return "기타"  # 정의 불명 문자열은 기타로


def ensure_gender_label(
    df_hybrid: pd.DataFrame,
    original_csv_path: str = "ecommerce_customer_data.csv",
    code_map_path: str = "gender_code_map.json",
) -> pd.DataFrame:
    """
    하이브리드 df에 GenderLabel 보장:
      1) 원본 CSV(ecommerce_customer_data.csv)의 Gender 문자열을 CustomerID로 조인해 표준 라벨 우선 사용
      2) 남은 결측은 숫자 코드→라벨 매핑으로 보완
    """
    df = df_hybrid.copy()

    # 1) 원본 조인 (CustomerID 기준)
    if os.path.exists(original_csv_path):
        try:
            raw = pd.read_csv(original_csv_path, usecols=["CustomerID", "Gender"])
            raw["GenderLabel_from_raw"] = raw["Gender"].map(_normalize_gender_text_to_label_ko)
            df = df.merge(raw[["CustomerID", "GenderLabel_from_raw"]], on="CustomerID", how="left")
        except Exception:
            df["GenderLabel_from_raw"] = np.nan
    else:
        df["GenderLabel_from_raw"] = np.nan

    # 2) 코드→라벨 매핑 로드(없으면 기본)
    code_map = DEFAULT_CODE_TO_LABEL_KO.copy()
    if os.path.exists(code_map_path):
        try:
            with open(code_map_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)  # {"1":"여성", "3":"남성", ...}
                code_map.update({int(k): v for k, v in loaded.items()})
        except Exception:
            pass

    # 3) 최종 GenderLabel 구성
    if "Gender" in df.columns:
        label_from_code = df["Gender"].map(code_map)
    else:
        label_from_code = pd.Series(index=df.index, dtype="object")

    df["GenderLabel"] = df["GenderLabel_from_raw"].fillna(label_from_code)
    df.drop(columns=["GenderLabel_from_raw"], inplace=True)
    df["GenderLabel"] = df["GenderLabel"].fillna("미상")

    return df


# -------------------------------
# CustomerID_clean 보장 (CSV에 CustomerID가 없어도 동작)
# -------------------------------
def ensure_customer_id_clean(df: pd.DataFrame) -> pd.DataFrame:
    """CustomerID_clean을 항상 보장한다.
    - CustomerID_clean이 있으면 결측/공백만 보완
    - CustomerID가 있으면 정리해서 생성
    - 둘 다 없으면 행 순서 기반으로 CUST00001~ 생성
    """
    out = df.copy()

    def _is_bad(v) -> bool:
        if pd.isna(v):
            return True
        s = str(v).strip()
        return (s == "") or (s.lower() in {"nan", "none", "nat", "null"})

    # 이미 있으면 결측만 보완
    if "CustomerID_clean" in out.columns:
        bad = out["CustomerID_clean"].map(_is_bad)
        if bad.any():
            fallback = pd.Series(np.arange(1, len(out) + 1), index=out.index).map(lambda i: f"CUST{i:05d}")
            out.loc[bad, "CustomerID_clean"] = fallback.loc[bad]
        return out

    # CustomerID로부터 생성
    if "CustomerID" in out.columns:
        def _clean_id(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            return np.nan if (s == "" or s.lower() in {"nan", "none", "nat", "null"}) else s
        out["CustomerID_clean"] = out["CustomerID"].map(_clean_id)
    else:
        out["CustomerID_clean"] = pd.Series(np.arange(1, len(out) + 1), index=out.index).map(lambda i: f"CUST{i:05d}")

    bad = out["CustomerID_clean"].map(_is_bad)
    if bad.any():
        fallback = pd.Series(np.arange(1, len(out) + 1), index=out.index).map(lambda i: f"CUST{i:05d}")
        out.loc[bad, "CustomerID_clean"] = fallback.loc[bad]

    return out


def clean_customer_type(x) -> str:
    """A:, B: 같은 접두어 제거해서 고객유형 라벨만 남긴다."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "미분류"
    s = str(x).strip()
    if ":" in s:
        left, right = s.split(":", 1)
        if len(left.strip()) <= 3:
            return right.strip()
    return s

# -------------------------------
# Data Loaders
# -------------------------------
@st.cache_data(show_spinner=False)
def load_main():
    try:
        df = pd.read_csv("ecommerce_customer_churn_hybrid_with_id.csv")
    except Exception:
        return None

    # ✅ CustomerID_clean 항상 보장
    df = ensure_customer_id_clean(df)

    # 성별 라벨 보장(원본 조인 + 코드 보완)
    df = ensure_gender_label(df)

    return df


@st.cache_data(show_spinner=False)
def load_featured():
    try:
        dff = pd.read_csv("ecommerce_customer_data_featured.csv")
    except Exception:
        return None

    # ✅ CustomerID_clean 항상 보장
    dff = ensure_customer_id_clean(dff)

    return dff


# actions.db 로드
@st.cache_data(show_spinner=False)
def load_actions():
    """actions.db에서 고객별 최근 액션 이력을 불러온다."""
    if not os.path.exists("actions.db"):
        return pd.DataFrame(columns=["customer_id", "action", "ts"])

    conn = sqlite3.connect("actions.db")
    df = pd.read_sql_query(
        "SELECT customer_id, action, ts FROM actions",
        conn
    )
    conn.close()

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


df = load_main()
dff = load_featured()
actions_df = load_actions()

# 데이터 로드 실패 방어
if df is None or (hasattr(df, "empty") and df.empty):
    st.error("메인 데이터(ecommerce_customer_churn_hybrid_with_id.csv)를 불러오지 못했습니다. 파일 경로/이름을 확인하세요.")
    st.stop()

if dff is None:
    dff = pd.DataFrame()

# -------------------------------
# Helpers
# -------------------------------
def exists(col):
    return col in df.columns


def col_or_none(cols):
    return [c for c in cols if c in df.columns]


def get_p99(series: pd.Series) -> float:
    try:
        p = float(series.quantile(0.99))
        return p if p > 0 else 1.0
    except Exception:
        return 1.0


def compute_risk_score_100(series: pd.Series) -> pd.Series:
    """모델 raw 점수를 0~100점 위험도로 변환 (상위% 기준)."""
    s = pd.to_numeric(series, errors="coerce")
    if not s.notna().any():
        return pd.Series(np.nan, index=series.index)
    ranks = s.rank(pct=True)  # 0~1, 값이 클수록 상위
    scores = (ranks * 100).round(0)
    return scores


def risk_level_from_score(score) -> str:
    """0~100 위험도 점수를 등급 텍스트로 변환."""
    if pd.isna(score):
        return "정보없음"
    v = float(score)
    if v >= 90:
        return "매우 높음"
    if v >= 70:
        return "높음"
    if v >= 40:
        return "보통"
    if v >= 20:
        return "낮음"
    return "매우 낮음"

# -------------------------------
# KPI 숫자 클릭 가능 CSS (모양은 그대로, 숫자 위에 투명 링크 오버레이)
# -------------------------------
st.markdown("""
<style>
.kpi-link { position: relative; display:block; top:-64px; height:56px; margin-bottom:-56px;
            z-index:100; cursor:pointer; }
.kpi-link:hover { background: rgba(0,0,0,0.02); }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 사이드바: 도움말 / 전역 필터 / 임계값
# -------------------------------
with st.sidebar:
    st.header("❓ 도움말 / 사용법")
    with st.expander("전역 필터 사용법"):
        st.markdown("""
        - **나이**: 범위를 좁힐수록 해당 연령대만 분석됩니다.
        - **성별**: 원본 문자열을 표준화한 `GenderLabel`(남성/여성/기타/응답거부/미상) 기준으로 필터합니다.
        - **리피트/프리미엄 플래그**: 1(예)/0(아니오)로 세분화합니다.
        """)
    with st.expander("임계값 튜닝이란?"):
        st.markdown("""
        - 모델 점수(IF: `IF_AnomalyScore`, AE: `AE_ReconError`)가 **임계값 이상**이면 '이탈'로 판단합니다.
        - **동적 임계값 사용**을 켜면 슬라이더로 임계값을 직접 조정합니다.
          - 슬라이더를 **낮추면** 더 많은 고객이 이탈로 **표시**됩니다(재현율↑, 정밀도↓).
          - 슬라이더를 **높이면** 더 **엄격**해집니다(정밀도↑, 재현율↓).
        - 이 모드에서는 `Both_ChurnFlag_dyn`(IF & AE 모두 만족)이 고신뢰 위험군으로 사용됩니다.
        """)

with st.sidebar:
    st.header("🔎 전역 필터")

    # Age
    if exists("Age"):
        age_min, age_max = int(np.nanmin(df["Age"])), int(np.nanmax(df["Age"]))
        sel_age = st.slider("나이", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    else:
        sel_age = None

    # Gender (표준 라벨 기반)
    if exists("GenderLabel"):
        gender_labels = sorted(pd.Series(df["GenderLabel"].dropna().unique()).tolist())
        sel_gender_labels = st.multiselect("성별", gender_labels, default=[])
    else:
        sel_gender_labels = []

    # Premium-like flag
    premium_flag_col = "RepeatAndPremiumFlag" if exists("RepeatAndPremiumFlag") else None
    if premium_flag_col:
        premium_opt = st.selectbox("리피트/프리미엄", ["전체", "예(1)", "아니오(0)"])
    else:
        premium_opt = "전체"

    st.markdown("---")
    st.subheader("⚙️ 임계값 튜닝(실험)")
    use_dynamic = st.toggle("동적 임계값 사용", value=False)

    if use_dynamic:
        if exists("IF_AnomalyScore"):
            if_thr_default = float(df["IF_AnomalyScore"].quantile(0.95))
            if_thr_min = float(df["IF_AnomalyScore"].quantile(0.90))
            if_thr_max = float(df["IF_AnomalyScore"].quantile(0.99))
            if_thr = st.slider("IF 임계값", min_value=float(if_thr_min), max_value=float(if_thr_max), value=float(if_thr_default))
        else:
            if_thr = None

        if exists("AE_ReconError"):
            ae_thr_default = float(df["AE_ReconError"].quantile(0.95))
            ae_thr_min = float(df["AE_ReconError"].quantile(0.90))
            ae_thr_max = float(df["AE_ReconError"].quantile(0.99))
            ae_thr = st.slider("AE 임계값", min_value=float(ae_thr_min), max_value=float(ae_thr_max), value=float(ae_thr_default))
        else:
            ae_thr = None
    else:
        if_thr = None
        ae_thr = None

# 리스트 페이지가 동일 조건을 사용하도록 세션에 저장
st.session_state["sel_age"] = sel_age
st.session_state["sel_gender_labels"] = sel_gender_labels
st.session_state["premium_opt"] = premium_opt
st.session_state["use_dynamic"] = use_dynamic
st.session_state["if_thr"] = if_thr
st.session_state["ae_thr"] = ae_thr

# -------------------------------
# 필터 적용
# -------------------------------
filtered = df.copy()
if sel_age:
    filtered = filtered[(filtered["Age"] >= sel_age[0]) & (filtered["Age"] <= sel_age[1])]

# 성별 라벨로 필터
if sel_gender_labels:
    filtered = filtered[filtered["GenderLabel"].isin(sel_gender_labels)]

if premium_flag_col and premium_opt != "전체":
    filtered = filtered[filtered[premium_flag_col] == (1 if premium_opt.startswith("예") else 0)]

# 동적 플래그
if use_dynamic and exists("IF_AnomalyScore") and exists("AE_ReconError"):
    filtered = filtered.copy()
    filtered["IF_ChurnFlag_dyn"] = (filtered["IF_AnomalyScore"] >= if_thr).astype(int)
    filtered["AE_ChurnFlag_dyn"] = (filtered["AE_ReconError"] >= ae_thr).astype(int)
    filtered["Both_ChurnFlag_dyn"] = (filtered["IF_ChurnFlag_dyn"] & filtered["AE_ChurnFlag_dyn"]).astype(int)
    flag_col = "Both_ChurnFlag_dyn"
else:
    flag_col = "Both_ChurnFlag" if exists("Both_ChurnFlag") else None

# -------------------------------
# 오늘 우선 연락해야 할 고객 계산
# -------------------------------
ACTIONS_LOOKBACK_DAYS = 7  # 최근 N일 기준
ACTIONS_BENEFIT_KEYWORDS = ["쿠폰", "혜택", "VIP"]  # 혜택/쿠폰 발송 키워드

if not actions_df.empty:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=ACTIONS_LOOKBACK_DAYS)
    recent_actions = actions_df[actions_df["ts"] >= cutoff].copy()

    # 연락 이력이 있다고 보는 고객 (현재는 actions 전체를 연락으로 간주)
    contacted_ids = set(
        recent_actions["customer_id"].dropna().astype(str)
    )

    # 혜택(쿠폰/혜택/VIP 포함) 이력이 있는 고객
    benefit_mask = recent_actions["action"].fillna("").str.contains(
        "|".join(ACTIONS_BENEFIT_KEYWORDS),
        case=False,
        na=False
    )
    benefit_ids = set(
        recent_actions.loc[benefit_mask, "customer_id"].dropna().astype(str)
    )
else:
    contacted_ids = set()
    benefit_ids = set()

# 1) 이탈 위험 + 최근 N일 연락 없는 고객
risky_no_contact = pd.DataFrame()
if "CustomerID_clean" in filtered.columns:
    risky_base = filtered.copy()
    if flag_col and (flag_col in risky_base.columns):
        risky_base = risky_base[risky_base[flag_col] == 1]

    risky_base = risky_base[risky_base["CustomerID_clean"].notna()].copy()
    risky_base["cid_key"] = risky_base["CustomerID_clean"].astype(str)

    mask_no_contact = ~risky_base["cid_key"].isin(contacted_ids)
    risky_no_contact = risky_base[mask_no_contact].copy()

    # 위험도 높은 순 정렬 (raw 점수 기준)
    if "ChurnRiskScore" in risky_no_contact.columns:
        risky_no_contact = risky_no_contact.sort_values("ChurnRiskScore", ascending=False)

# 2) VIP 후보 + 최근 N일 혜택 미발송 고객
vip_no_benefit = pd.DataFrame()
if "CustomerID_clean" in filtered.columns:
    try:
        vip_score_df = compute_vip_propensity_score(filtered, ref_df=filtered)
        tmp = filtered.merge(
            vip_score_df[["CustomerID_clean", "VIP잠재지수"]],
            on="CustomerID_clean",
            how="left"
        )
        tmp = tmp[tmp["CustomerID_clean"].notna()].copy()
        tmp["cid_key"] = tmp["CustomerID_clean"].astype(str)

        VIP_THR = 80.0  # VIP 후보 기준 점수
        vip_base = tmp[tmp["VIP잠재지수"] >= VIP_THR].copy()

        mask_no_benefit = ~vip_base["cid_key"].isin(benefit_ids)
        vip_no_benefit = vip_base[mask_no_benefit].copy()

        vip_no_benefit = vip_no_benefit.sort_values("VIP잠재지수", ascending=False)
    except Exception:
        vip_no_benefit = pd.DataFrame()

# 오늘 보여줄 Top N (단, metric은 실제 건수 기반)
RISKY_TODAY_LIMIT = 10
VIP_TODAY_LIMIT = 7

risky_today_n = int(min(RISKY_TODAY_LIMIT, len(risky_no_contact)))
vip_today_n   = int(min(VIP_TODAY_LIMIT, len(vip_no_benefit)))

# -------------------------------
# Layout
# -------------------------------
st.title("🧭 고객 이탈 위험 대시보드")

# 필터 요약
filter_badges = []
if sel_age:
    filter_badges.append(f"나이 {sel_age[0]}~{sel_age[1]}세")
if sel_gender_labels:
    filter_badges.append("성별: " + ", ".join(sel_gender_labels))
if premium_flag_col and premium_opt != "전체":
    filter_badges.append(f"리피트/프리미엄: {premium_opt}")

if filter_badges:
    st.caption("현재 적용된 필터: " + " · ".join(filter_badges))
else:
    st.caption("현재 적용된 필터: 전체 고객")

tabs = st.tabs(["📊 개요", "🔍 고객 조회"])

# =========================================
# 📊 개요 탭
# =========================================
with tabs[0]:
    # 오늘 우선 관리해야 할 고객 요약 박스
    st.markdown("### 📌 우선 관리 고객")
    st.caption("금일 기준으로 연락·혜택 발송이 필요한 주요 고객 수입니다.")
    cc1, cc2 = st.columns(2)
    cc1.metric(
        "이탈 위험 + 최근 7일 연락 이력 없음",
        f"{risky_today_n}명"
    )
    cc2.metric(
        "VIP 후보 + 최근 7일 혜택 미발송",
        f"{vip_today_n}명"
    )
    st.caption("※ 현재 화면의 필터(나이/성별/리피트/임계값)와 최근 7일 기준으로 계산됩니다.")

    # 요약표용 CSS: 가로 스크롤 + 헤더/셀 줄바꿈 없음
    st.markdown(
        """
<style>
.today-summary-wrap {
  overflow-x: auto;
}
.today-summary-table {
  border-collapse: collapse;
  width: auto !important;
  table-layout: auto;
}
.today-summary-table th,
.today-summary-table td {
  padding: 8px 12px !important;
  white-space: nowrap;
  vertical-align: middle;
  font-size: 0.9rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # 상세 리스트(expander)
    with st.expander("우선 관리 대상 자세히 보기", expanded=False):
        left, right = st.columns(2)

        # ----- 이탈 위험 고객 -----
        with left:
            st.markdown("**이탈 위험 + 최근 7일 연락 이력 없음**")
            st.caption("최근 7일 동안 별도 연락이 없었고, 이탈 위험 점수가 높은 순으로 정렬된 고객입니다.")
            if risky_today_n == 0:
                st.write("해당 조건의 고객이 없습니다.")
            else:
                r_view = risky_no_contact.head(RISKY_TODAY_LIMIT).copy()
                r_view = r_view[r_view["CustomerID_clean"].notna()].copy()

                # 0~100 위험도 + 등급 계산
                if "ChurnRiskScore" in r_view.columns:
                    r_view["RiskScore100"] = compute_risk_score_100(r_view["ChurnRiskScore"])
                    r_view["RiskLevel"] = r_view["RiskScore100"].apply(risk_level_from_score)

                # 링크 컬럼
                r_view["고객ID"] = r_view["CustomerID_clean"].apply(
                    lambda cid: f"<a href='/{DETAIL_PAGE_SLUG}?customer_id={quote(str(cid))}' target='_self'>{cid}</a>"
                )
                base_cols = [
                    "고객ID",
                    "RiskLevel",
                    "RiskScore100",
                    "PurchaseFrequency",
                    "CSFrequency",
                ]
                cols = ["고객ID"] + [c for c in base_cols if c in r_view.columns and c != "고객ID"]
                r_view = r_view[cols]
                r_view = rename_for_display(r_view)

                fmt_r = {}
                if "이탈 위험 점수(0~100)" in r_view.columns:
                    fmt_r["이탈 위험 점수(0~100)"] = "{:.0f}"
                for c in ["구매 빈도(월 평균)", "상담 빈도(월 평균)"]:
                    if c in r_view.columns:
                        fmt_r[c] = "{:.2f}"

                styler_r = (
                    r_view.style
                    .hide(axis="index")
                    .format(fmt_r)
                    .set_table_attributes('class="today-summary-table"')
                )
                html_r = styler_r.to_html(escape=False)
                st.markdown(f"<div class='today-summary-wrap'>{html_r}</div>", unsafe_allow_html=True)

        # ----- VIP 전환 후보 -----
        with right:
            st.markdown("**VIP 후보 + 최근 7일 혜택 미발송**")
            st.caption("VIP로 성장 가능성이 높고, 최근 7일 동안 별도 혜택이 발송되지 않은 고객입니다.")
            if vip_today_n == 0:
                st.write("해당 조건의 고객이 없습니다.")
            else:
                v_view = vip_no_benefit.head(VIP_TODAY_LIMIT).copy()
                v_view = v_view[v_view["CustomerID_clean"].notna()].copy()
                v_view["고객ID"] = v_view["CustomerID_clean"].apply(
                    lambda cid: f"<a href='/{DETAIL_PAGE_SLUG}?customer_id={quote(str(cid))}' target='_self'>{cid}</a>"
                )
                base_cols_v = [
                    "고객ID",
                    "VIP잠재지수",
                    "CustomerLifetimeValue",
                    "PurchaseFrequency",
                    "AverageOrderValue",
                    "TotalEngagementScore",
                    "EmailEngagementRate",
                    "MobileAppUsage",
                ]
                cols_v = ["고객ID"] + [c for c in base_cols_v if c in v_view.columns and c != "고객ID"]
                v_view = v_view[cols_v]
                v_view = rename_for_display(v_view)
                fmt_v = {
                    "VIP전환지수": "{:.0f}",
                    "고객생애가치": "{:,.0f}",
                    "구매 빈도(월 평균)": "{:.2f}",
                    "평균주문금액": "{:,.0f}",
                    "총참여점수": "{:.2f}",
                    "이메일참여율": "{:.2f}",
                    "모바일앱사용": "{:.0f}",
                }
                styler_v = (
                    v_view.style
                    .hide(axis="index")
                    .format(fmt_v)
                    .set_table_attributes('class="today-summary-table"')
                )
                html_v = styler_v.to_html(escape=False)
                st.markdown(f"<div class='today-summary-wrap'>{html_v}</div>", unsafe_allow_html=True)

    # KPI-구분선-제목 사이 여백 조정 (줄을 위로, 제목과는 여백 확보)
    st.markdown(
        "<hr style='margin-top:8px; margin-bottom:22px; opacity:0.22;'>",
        unsafe_allow_html=True
    )


    # -------------------------------
    # 🧩 고객유형 분포 (전역 필터 반영)
    # -------------------------------
    if ("BehaviorClusterName" in filtered.columns) or ("BehaviorCluster" in filtered.columns):
        cluster_col = "BehaviorClusterName" if "BehaviorClusterName" in filtered.columns else "BehaviorCluster"
        tmp = filtered.copy()
        tmp["고객유형"] = tmp[cluster_col].map(clean_customer_type)

        # 평균 이탈 위험 점수(0~100)
        if "ChurnRiskScore" in tmp.columns:
            tmp["_risk100"] = compute_risk_score_100(tmp["ChurnRiskScore"])
        else:
            tmp["_risk100"] = np.nan

        # 고신뢰 이탈 플래그(동적 임계값 우선)
        churn_flag = None
        if flag_col and (flag_col in tmp.columns):
            churn_flag = flag_col
        elif "Both_ChurnFlag" in tmp.columns:
            churn_flag = "Both_ChurnFlag"

        if churn_flag:
            tmp["_high_churn"] = (tmp[churn_flag] == 1).astype(int)
        else:
            tmp["_high_churn"] = 0

        dist = (
            tmp.groupby("고객유형", dropna=False)
            .agg(**{
                "고객 수": ("고객유형", "size"),
                "평균 이탈 위험 점수(0~100)": ("_risk100", "mean"),
                "고신뢰 이탈(%)": ("_high_churn", "mean"),
            })
            .reset_index()
        )
        dist["비중(%)"] = dist["고객 수"] / max(1, len(tmp)) * 100.0
        dist["고신뢰 이탈(%)"] = dist["고신뢰 이탈(%)"] * 100.0

        # 이탈율 높은 순
        dist = dist[["고객유형", "고객 수", "비중(%)", "평균 이탈 위험 점수(0~100)", "고신뢰 이탈(%)"]]
        dist = dist.sort_values("고신뢰 이탈(%)", ascending=False, na_position="last").reset_index(drop=True)
        dist.index = np.arange(1, len(dist) + 1)  

        st.markdown("### 🧩 고객유형 분포")
        st.caption("전역 필터가 반영된 분포입니다. (고객유형 칸을 클릭하면 해당 유형 고객 목록으로 이동합니다.)")

        tbl_nonce = st.session_state.get("_customer_type_tbl_nonce", 0)
        _table_key = f"customer_type_table_{tbl_nonce}"

        # ✅ 변경: 체크박스(행 선택) 대신 셀 선택(single-cell)로 이동
        event = None
        try:
            event = st.dataframe(
                dist,
                use_container_width=True,
                hide_index=False,
                on_select="rerun",
                selection_mode="single-cell",
                key=_table_key,
                column_config={
                    "비중(%)": st.column_config.NumberColumn("비중(%)", format="%.1f%%"),
                    "평균 이탈 위험 점수(0~100)": st.column_config.NumberColumn("평균 이탈 위험 점수(0~100)", format="%.0f"),
                    "고신뢰 이탈(%)": st.column_config.NumberColumn("고신뢰 이탈(%)", format="%.1f%%"),
                },
            )
        except Exception:
            # 셀 선택 미지원 환경이면(구버전) 표만 보여주고 체크박스는 없게 유지
            st.dataframe(
                dist,
                use_container_width=True,
                hide_index=False,
                column_config={
                    "비중(%)": st.column_config.NumberColumn("비중(%)", format="%.1f%%"),
                    "평균 이탈 위험 점수(0~100)": st.column_config.NumberColumn("평균 이탈 위험 점수(0~100)", format="%.0f"),
                    "고신뢰 이탈(%)": st.column_config.NumberColumn("고신뢰 이탈(%)", format="%.1f%%"),
                },
            )
            event = None

        # 선택된 셀 → 해당 행의 고객유형 페이지로 이동
        sel_cells = []
        try:
            sel_cells = list(event.selection.cells) if event is not None else []
        except Exception:
            sel_cells = []

        if sel_cells:
            ridx = int(sel_cells[0][0])  # (row_position, column_name)
            sel_type = str(dist.iloc[ridx]["고객유형"]).strip()

            st.session_state["selected_customer_type"] = sel_type
            st.session_state["_customer_type_tbl_nonce"] = tbl_nonce + 1

            if os.path.exists("pages/04_Customer_Type_List.py"):
                st.switch_page("pages/04_Customer_Type_List.py")
            else:
                st.warning("pages/04_Customer_Type_List.py 파일이 없어 이동할 수 없습니다. (pages 폴더에 파일을 추가하세요.)")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    # 전체 이탈 위험 현황 요약
    st.subheader("📊 전체 이탈 위험 현황 요약")
    st.caption("이탈 위험 고객 수를 유형별로 나눈 요약입니다.")

    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(filtered)
    churn_if = int(filtered["IF_ChurnFlag"].sum()) if exists("IF_ChurnFlag") else 0
    churn_ae = int(filtered["AE_ChurnFlag"].sum()) if exists("AE_ChurnFlag") else 0
    churn_both = int(filtered[flag_col].sum()) if flag_col else 0

    col1.metric("총 고객 수(필터 반영)", f"{total_customers:,}")
    with col2:
        st.metric("이상행동 기반 이탈 의심 고객 수", f"{churn_if:,}")
        st.markdown("<a class='kpi-link' href='/Risky_List?src=if' title='이상행동 기반 이탈 고객 목록'></a>", unsafe_allow_html=True)
    with col3:
        st.metric("패턴 변화 기반 이탈 의심 고객 수", f"{churn_ae:,}")
        st.markdown("<a class='kpi-link' href='/Risky_List?src=ae' title='패턴 변화 기반 이탈 고객 목록'></a>", unsafe_allow_html=True)
    with col4:
        ratio = churn_both/total_customers*100 if total_customers else 0
        col4.metric("두 기준 모두 위험한 고위험 고객 수", f"{churn_both:,} ({ratio:.2f}%)")
        st.markdown("<a class='kpi-link' href='/Risky_List?src=both' title='고위험 이탈 고객 목록'></a>", unsafe_allow_html=True)

    # 이탈 위험 고객 리스트
    st.subheader("🚨 이탈 위험 고객 리스트")
    st.caption("이탈 위험 점수가 높은 순으로 정렬된 고객입니다. 고객ID를 클릭하면 상세 화면으로 이동합니다.")
    top_k = st.slider("리스트 크기", min_value=5, max_value=200, value=10, step=5)

    list_df = filtered.copy()
    if flag_col:
        list_df = list_df[list_df[flag_col] == 1]

    # 고객ID 없는 행 제거
    if "CustomerID_clean" in list_df.columns:
        list_df = list_df[list_df["CustomerID_clean"].notna()]
    elif "CustomerID" in list_df.columns:
        list_df = list_df[list_df["CustomerID"].notna()]

    # 위험도 기준 정렬
    if "ChurnRiskScore" in list_df.columns:
        list_df = list_df.sort_values("ChurnRiskScore", ascending=False)
        # 0~100 위험도 + 등급 계산
        list_df["RiskScore100"] = compute_risk_score_100(list_df["ChurnRiskScore"])
        list_df["RiskLevel"] = list_df["RiskScore100"].apply(risk_level_from_score)

    # 표에 넣을 컬럼
    base_cols = [
        "CustomerID_clean",
        "GenderLabel",
        "RiskLevel",
        "RiskScore100",
        "PurchaseFrequency",
        "CSFrequency",
        "AverageSatisfactionScore",
        "NegativeExperienceIndex",
        "EmailEngagementRate",
        "TotalEngagementScore",
    ]
    cols_to_show = [c for c in base_cols if c in list_df.columns]

    risky_customers = list_df.head(top_k)[cols_to_show].copy()

    if risky_customers.empty:
        st.info("현재 조건에서 표시할 고객이 없습니다.")
    else:
        # 순위 + 고객ID 링크
        risky_customers.insert(0, "", np.arange(1, len(risky_customers) + 1))
        id_col = "CustomerID_clean" if "CustomerID_clean" in risky_customers.columns else ("CustomerID" if "CustomerID" in risky_customers.columns else None)
        if id_col:
            risky_customers["고객ID"] = risky_customers[id_col].apply(
                lambda cid: f"<a href='/{DETAIL_PAGE_SLUG}?customer_id={quote(str(cid))}' target='_self'>{cid}</a>"
            )
        else:
            risky_customers["고객ID"] = "-"

        # 화면 표시용 DF (CustomerID_clean 제거 + 한글 라벨)
        display_df = risky_customers.drop(columns=["CustomerID_clean", "CustomerID"], errors="ignore")
        display_df = rename_for_display(display_df)

        risk_score_label = KOR_COL.get("RiskScore100", "RiskScore100")
        risk_level_label = KOR_COL.get("RiskLevel", "RiskLevel")

        # 표시 순서: 순위 → 고객ID → 위험 수준 → 이탈 위험 점수 → 나머지
        display_cols = ["", "고객ID"]
        if risk_level_label in display_df.columns:
            display_cols.append(risk_level_label)
        if risk_score_label in display_df.columns:
            display_cols.append(risk_score_label)
        display_cols += [c for c in display_df.columns if c not in display_cols]

        # 포맷
        fmt_map = {
            risk_score_label: "{:.0f}",
            "구매 빈도(월 평균)": "{:.2f}",
            "상담 빈도(월 평균)": "{:.2f}",
            "평균만족도": "{:.2f}",
            "부정경험지수": "{:.2f}",
            "이메일참여율": "{:.2f}",
            "총참여점수": "{:.2f}",
        }

        styler = (
            display_df[display_cols]
            .style
            .format({k: v for k, v in fmt_map.items() if k in display_df.columns})
            .hide(axis="index")
            .set_table_attributes('id="risky_table" class="dataframe"')
        )

        # 위험도(100점)에 색 농도 주기
        def style_risk(series: pd.Series):
            if series.name != risk_score_label:
                return [""] * len(series)
            vals = pd.to_numeric(series, errors="coerce")
            if vals.notna().any():
                vmin = float(vals.min(skipna=True))
                vmax = float(vals.max(skipna=True))
            else:
                vmin, vmax = 0.0, 1.0
            rng = (vmax - vmin) if vmax > vmin else 1.0
            alphas = 0.15 + 0.75 * (vals - vmin) / rng
            alphas = alphas.clip(lower=0, upper=1).fillna(0)
            return [f"background-color: rgba(255,0,0,{a:.2f})" for a in alphas]

        if risk_score_label in display_df.columns:
            styler = styler.apply(style_risk, axis=0)

        # 표 가로 스크롤 + 헤더/셀 줄바꿈 없음
        st.markdown(
            """
<style>
.risky-wrap {
  overflow-x: auto;
}
#risky_table {
  border-collapse: collapse;
  width: auto !important;
  table-layout: auto;
}
#risky_table th, #risky_table td {
  padding: 10px 12px !important;
  line-height: 1.45;
  vertical-align: middle;
  white-space: nowrap;
}
</style>
""",
            unsafe_allow_html=True,
        )

        html_main = styler.to_html(escape=False)
        st.markdown(f"<div class='risky-wrap'>{html_main}</div>", unsafe_allow_html=True)

        # CSV 다운로드
        export_df = display_df[display_cols].copy()
        export_df.rename(columns={"": "순위"}, inplace=True)
        if "고객ID" in export_df.columns and "CustomerID" not in export_df.columns:
            export_df.insert(1, "CustomerID", export_df["고객ID"].str.extract(r'>(.*?)<')[0])
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 리스트 내려받기 (CSV)",
            data=csv_bytes,
            file_name="risky_customers.csv",
            mime="text/csv",
        )

    st.markdown("---")
    # 부가 요약 (일부 피처) — 표 머리만 한글
    if dff is not None:
        st.subheader("📈 요약 통계 (일부 피처)")
        st.caption("주요 지표의 분포를 요약한 표입니다. 상위/하위 분위수 확인에 활용할 수 있습니다.")
        sample_cols = [c for c in [
            "Age", "TotalPurchases", "AverageOrderValue", "CustomerLifetimeValue",
            "EmailEngagementRate", "MobileAppUsage", "CustomerServiceInteractions",
            "AverageSatisfactionScore", "ChurnRiskScore"
        ] if c in dff.columns]
        if sample_cols:
            desc = dff[sample_cols].describe().T
            desc = rename_for_display(desc)
            st.dataframe(desc, use_container_width=True)

# =========================================
# 고객 조회 탭
# =========================================
with tabs[1]:
    st.subheader("고객 ID로 조회")
    st.caption("특정 고객ID를 직접 입력해 해당 고객의 상세 정보를 확인할 수 있습니다.")
    cid = st.text_input("CustomerID 입력", value="")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("상세 페이지 열기"):
            if cid:
                page_href = f"/{DETAIL_PAGE_SLUG}?customer_id={quote(str(cid))}"
                st.markdown(f"[👉 고객 상세 페이지로 이동]({page_href})")
            else:
                st.warning("CustomerID를 입력하세요.")

    with colB:
        if cid:
            q = df[df.get("CustomerID_clean") == str(cid)]
            if not q.empty and "ChurnRiskScore" in df.columns:
                p99 = get_p99(df["ChurnRiskScore"])
                risk = float(q.iloc[0]["ChurnRiskScore"]) / p99
                risk = min(max(risk, 0.0), 1.0)
                st.write("해당 고객의 상대적 이탈 위험도(상위 % 기준):")
                st.progress(risk)
                st.dataframe(rename_for_display(q.head(1)).T, use_container_width=True)
            elif q.empty:
                st.info("일치하는 고객이 없습니다.")