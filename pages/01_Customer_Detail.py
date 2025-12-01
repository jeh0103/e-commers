# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, datetime, json, os, math
from urllib.parse import unquote

st.set_page_config(page_title="👤 고객 상세", layout="wide")

# -------------------------------
# 공통 컬럼 한글 라벨 맵
# -------------------------------
COL_LABEL_KO = {
    "CustomerID": "고객ID",
    "CustomerID_clean": "고객ID",
    "GenderLabel": "성별",
    "Age": "나이",
    "IncomeLevel": "소득 수준",
    "CustomerTenure": "이용 개월(추정)",
    "RepeatCustomer": "재구매 고객 코드",
    "RepeatAndPremiumFlag": "리피트/프리미엄 플래그",

    "TotalPurchases": "총 구매 횟수",
    "AverageOrderValue": "평균 주문 금액",
    "PurchaseFrequency": "구매 빈도",
    "AvgPurchaseInterval": "평균 구매 간격",
    "CSFrequency": "상담 빈도",
    "AverageSatisfactionScore": "평균 만족도",
    "NegativeExperienceIndex": "부정 경험 지수",
    "EmailEngagementRate": "이메일 참여율",
    "TotalEngagementScore": "총 참여 점수",
    "RecencyProxy": "휴면 징후 지수",

    "ChurnRiskScore": "이탈 위험 점수",
    "IF_AnomalyScore": "이상행동 점수(IF)",
    "AE_ReconError": "패턴 차이 점수(AE)",
}

def col_label(col: str) -> str:
    return COL_LABEL_KO.get(col, col)

# -------------------------------
# Query-param helpers
# -------------------------------
def qp_get(name: str):
    try:
        v = st.query_params.get(name)
    except Exception:
        v = st.experimental_get_query_params().get(name)
    if isinstance(v, list):
        v = v[0] if v else None
    return v

# -------------------------------
# Gender label helpers
# -------------------------------
GENDER_CODE_MAP_PATH = "gender_code_map.json"   # 코드→라벨(기본값 보완/일괄 지정)
GENDER_LABEL_MAP_PATH = "gender_label_map.json" # 관리자 커스텀 라벨 저장(코드→라벨)

DEFAULT_CODE_TO_LABEL_KO = {
    1: "여성",
    3: "남성",
    5: "응답거부",
    4: "기타/미상",
    2: "남성",
    0: "여성",
}

def _code_key(v):
    try:
        f = float(v); i = int(f)
        return str(i) if f == i else str(f)
    except Exception:
        return str(v)

def _normalize_gender_text_to_label_ko(x) -> str:
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
    return "기타"

def _load_json(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def ensure_gender_label(df: pd.DataFrame,
                        original_csv_path: str = "ecommerce_customer_data.csv") -> pd.DataFrame:
    """
    GenderLabel 생성 우선순위:
    1) 원본 CSV(ecommerce_customer_data.csv)의 Gender 문자열을 CustomerID로 조인 후 표준화
    2) 결측은 코드→라벨 맵으로 보완 (기본맵 + gender_code_map.json + gender_label_map.json + 세션)
    """
    out = df.copy()

    # 1) 원본 문자열 조인
    if os.path.exists(original_csv_path):
        try:
            raw = pd.read_csv(original_csv_path, usecols=["CustomerID", "Gender"])
            raw["GenderLabel_from_raw"] = raw["Gender"].map(_normalize_gender_text_to_label_ko)
            out = out.merge(raw[["CustomerID", "GenderLabel_from_raw"]], on="CustomerID", how="left")
        except Exception:
            out["GenderLabel_from_raw"] = np.nan
    else:
        out["GenderLabel_from_raw"] = np.nan

    # 2) 코드→라벨 맵 구성
    code_map = DEFAULT_CODE_TO_LABEL_KO.copy()
    code_json = _load_json(GENDER_CODE_MAP_PATH)
    if code_json:
        try:
            code_map.update({int(k): v for k, v in code_json.items()})
        except Exception:
            pass
    label_json = _load_json(GENDER_LABEL_MAP_PATH)
    if label_json:
        try:
            code_map.update({int(k): v for k, v in label_json.items()})
        except Exception:
            pass
    if "gender_label_map" in st.session_state and isinstance(st.session_state["gender_label_map"], dict):
        try:
            code_map.update({int(k): v for k, v in st.session_state["gender_label_map"].items()})
        except Exception:
            pass

    # 숫자 코드에서 라벨 생성
    if "Gender" in out.columns:
        label_from_code = out["Gender"].map(code_map)
    else:
        label_from_code = pd.Series(index=out.index, dtype="object")

    # 최종 라벨
    out["GenderLabel"] = out["GenderLabel_from_raw"].fillna(label_from_code)
    out.drop(columns=["GenderLabel_from_raw"], inplace=True)
    out["GenderLabel"] = out["GenderLabel"].fillna("미상")
    return out

# -------------------------------
# Data
# -------------------------------
@st.cache_data(show_spinner=False)
def load_main():
    df = pd.read_csv("ecommerce_customer_churn_hybrid_with_id.csv")
    if "CustomerID" in df.columns:
        def _clean(x):
            if pd.isna(x): return np.nan
            s = str(x).strip()
            return np.nan if (s == "" or s.lower() in {"nan", "none", "nat", "null"}) else s
        df["CustomerID_clean"] = df["CustomerID"].map(_clean)

    # 성별 라벨 보장
    df = ensure_gender_label(df)
    return df

df = load_main()

def exists(c): return c in df.columns
def p99(x: pd.Series) -> float:
    try:
        q = float(x.quantile(0.99))
        return q if q > 0 else 1.0
    except Exception:
        return 1.0

# -------------------------------
# Read target customer
# -------------------------------
customer_id = qp_get("customer_id")
if not customer_id:
    st.error("customer_id 파라미터가 없습니다. 대시보드에서 고객ID를 클릭해 오세요.")
    st.stop()

customer_id = unquote(customer_id)
row_df = df[df.get("CustomerID_clean") == str(customer_id)]
if row_df.empty:
    st.error("해당 CustomerID가 존재하지 않습니다.")
    st.stop()

row = row_df.iloc[0]

# -------------------------------
# Header
# -------------------------------
try:
    st.page_link("app_enhanced.py", label="← 대시보드로", icon="🏠")
except Exception:
    st.markdown("[← 대시보드로](/)")

st.title("👤 고객 상세")
st.caption(f"고객ID: {customer_id}")

# -------------------------------
# Key-Value table helper
# -------------------------------
def kv_table(pairs):
    df_kv = pd.DataFrame(pairs, columns=["항목", "값"])
    def _fmt(x):
        if isinstance(x, (int, np.integer)): return f"{x:,}"
        if isinstance(x, (float, np.floating)):
            return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"
        return x
    df_kv["값"] = df_kv["값"].map(_fmt)
    return df_kv

# -------------------------------
# 기본 정보 / 활동·만족 지표
# -------------------------------
colL, colR = st.columns([1,1])

with colL:
    st.subheader("기본 정보")
    age_int = int(np.round(row["Age"])) if exists("Age") and pd.notna(row["Age"]) else None
    sex_label = row.get("GenderLabel", None)

    base_pairs = [("고객ID", customer_id)]
    if sex_label is not None:
        base_pairs.append(("성별", sex_label))
    if age_int is not None:
        base_pairs.append(("나이", age_int))
    for c, label in [
        ("IncomeLevel", "소득 수준"),
        ("CustomerTenure", "이용 개월(추정)"),
        ("RepeatCustomer", "재구매 고객 코드"),
        ("RepeatAndPremiumFlag", "리피트/프리미엄 플래그"),
    ]:
        if exists(c): base_pairs.append((label, row[c]))
    st.table(kv_table(base_pairs))

with colR:
    st.subheader("활동/만족 지표")
    feature_candidates = [
        "TotalPurchases","AverageOrderValue","PurchaseFrequency","AvgPurchaseInterval",
        "CSFrequency","AverageSatisfactionScore","NegativeExperienceIndex",
        "EmailEngagementRate","TotalEngagementScore","RecencyProxy"
    ]
    feat_cols = [c for c in feature_candidates if exists(c)]
    # 컬럼 이름 → 한글 라벨로 변환
    feat_pairs = [(col_label(c), row[c]) for c in feat_cols]
    st.table(kv_table(feat_pairs))

# -------------------------------
# Risk Gauge (p99 scaling) & Churn type
# -------------------------------
st.markdown("---")
g1, g2 = st.columns([2,1])

with g1:
    st.subheader("🚨 이탈 위험도")
    if exists("ChurnRiskScore"):
        scale = p99(df["ChurnRiskScore"])
        val = float(row["ChurnRiskScore"])
        meter = min(max(val/scale, 0.0), 1.0)
        st.progress(meter)
        st.caption(f"현재 점수: {val:.2f} / 상위 1% 기준 점수: {scale:.2f}")
    else:
        st.info("이탈 위험 점수(ChurnRiskScore) 컬럼이 없어 게이지를 표시할 수 없습니다.")

with g2:
    st.subheader("상태 요약")
    if all(exists(c) for c in ["Both_ChurnFlag","IF_ChurnFlag","AE_ChurnFlag"]):
        if int(row["Both_ChurnFlag"]) == 1:
            st.error("고신뢰 이탈 고객 (이상행동·패턴변화 모두 위험)")
        elif int(row["IF_ChurnFlag"]) == 1:
            st.warning("불만/이상행동 기반 이탈 신호")
        elif int(row["AE_ChurnFlag"]) == 1:
            st.info("이용 패턴 감소 기반 이탈 신호")
        else:
            st.success("특이 신호 없음")
    else:
        st.caption("이탈 플래그 컬럼이 없어 상태를 계산할 수 없습니다.")

# -------------------------------
# 활동/만족 지표 - 전체 대비 분위 & 리스크 시각화
# -------------------------------
st.markdown("---")
st.subheader("📊 전체 고객 대비 위치 & 리스크(%)")

RISK_DIR = {
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

def percentile(series: pd.Series, v):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or pd.isna(v):
        return np.nan
    return float((s <= float(v)).sum()) / float(len(s))

rows_feat = []
for c in feat_cols:
    val = float(row[c]) if pd.notna(row[c]) else np.nan
    pct = percentile(df[c], val)
    dirc = RISK_DIR.get(c, "neutral")
    if np.isnan(pct):
        risk = np.nan
    else:
        if dirc == "higher_worse":
            risk = pct
        elif dirc == "lower_worse":
            risk = 1.0 - pct
        else:
            risk = abs(pct - 0.5) * 2
    rows_feat.append({
        "지표": col_label(c),
        "값": val,
        "분위(%)": None if np.isnan(pct) else pct*100.0,
        "리스크(%)": None if np.isnan(risk) else risk*100.0,
        "리스크 방향": {"higher_worse":"↑ 높을수록 위험", "lower_worse":"↓ 낮을수록 위험"}.get(dirc, "중립"),
    })

feat_view = pd.DataFrame(rows_feat)
fmt_cols = { "값":"{:.2f}", "분위(%)":"{:,.0f}%", "리스크(%)":"{:,.0f}%" }
for k, f in list(fmt_cols.items()):
    if k not in feat_view.columns: del fmt_cols[k]

def style_red_percent(series: pd.Series):
    if series.name != "리스크(%)":
        return [""] * len(series)
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
    vmax = float(vals.max()) if len(vals) else 100.0
    vmax = max(vmax, 1.0)
    styles = []
    for v in vals:
        a = 0.15 + 0.75 * (float(v) / vmax)
        a = max(0, min(1, a))
        styles.append(f"background-color: rgba(255,0,0,{a:.2f})")
    return styles

styler_feat = feat_view.style.format(fmt_cols).hide(axis="index").apply(style_red_percent, axis=0)
st.markdown(styler_feat.to_html(escape=False), unsafe_allow_html=True)

# -------------------------------
# 주요 위험 요인 & 관리자 지침
# -------------------------------
st.markdown("---")
st.subheader("🔥 Top 리스크 요인 & 즉시 액션")

candidate_cols = [
    "PurchaseFrequency", "CSFrequency", "RecencyProxy",
    "AverageSatisfactionScore", "NegativeExperienceIndex",
    "EmailEngagementRate", "TotalEngagementScore",
]
driver_cols = [c for c in candidate_cols if exists(c)]

drivers = None
if driver_cols and exists("Both_ChurnFlag"):
    healthy = df[df["Both_ChurnFlag"] == 0][driver_cols].copy()
    mu = healthy.mean(numeric_only=True)
    sigma = healthy.std(numeric_only=True).replace(0, 1e-6)
    z = ((row[driver_cols] - mu) / sigma).astype(float)
    # 크기 기준으로 정렬 (절대값 큰 순)
    drivers = z.sort_values(key=lambda s: s.abs(), ascending=False)

    NAME = {
        "PurchaseFrequency":      "구매 빈도",
        "CSFrequency":            "상담 빈도",
        "RecencyProxy":           "활동저하 지수",
        "AverageSatisfactionScore": "만족도",
        "NegativeExperienceIndex":  "부정경험 지수",
        "EmailEngagementRate":      "이메일 참여율",
        "TotalEngagementScore":     "총 참여 점수",
    }

    # 위험 방향 정의(위에 있는 RISK_DIR와 동일한 의미)
    DIR = {
        "CSFrequency":            "higher_worse",
        "RecencyProxy":           "higher_worse",
        "NegativeExperienceIndex":"higher_worse",
        "PurchaseFrequency":      "lower_worse",
        "AverageSatisfactionScore":"lower_worse",
        "EmailEngagementRate":    "lower_worse",
        "TotalEngagementScore":   "lower_worse",
    }

    def is_bad(feat: str, zval: float) -> bool:
        d = DIR.get(feat, "neutral")
        if d == "higher_worse":   # 값이 높을수록 위험
            return zval > 0.8
        if d == "lower_worse":    # 값이 낮을수록 위험
            return zval < -0.8
        return abs(zval) >= 1.5   # 방향 없는 경우

    def severity_badge_and_text(zval: float):
        sev = abs(float(zval))
        if sev >= 2.5:
            return "🔴 영향 큼", "매우 큼"
        elif sev >= 1.5:
            return "🟠 영향 중간", "큼"
        elif sev >= 1.0:
            return "🟡 영향 약함", "보통"
        else:
            return "⚪ 영향 거의 없음", "작음"

    def describe_problem_action(feat: str, zval: float):
        """비개발자용 문제 요약 + 권장 액션."""
        d = DIR.get(feat, "neutral")
        up = zval > 0  # 평균보다 높은지/낮은지

        # 기본값 (혹시 누락될 경우)
        problem = "정상 고객과 다른 패턴을 보입니다."
        action  = "상세 이력을 보고 원인을 파악한 뒤 맞춤 케어를 진행합니다."

        if feat == "CSFrequency":
            if up:
                problem = "상담 요청이 정상 고객보다 매우 자주 발생합니다."
                action  = "시니어 상담을 배정해 최근 이슈를 정리하고, 불만 원인을 바로 해소합니다."
            else:
                problem = "상담 요청은 많지 않지만, 이력 확인이 필요합니다."
                action  = "최근 주문·클레임 기록을 점검해 잠재 이슈를 사전에 차단합니다."

        elif feat == "RecencyProxy":
            if up:
                problem = "최근 접속·구매가 크게 줄어 휴면에 가까운 상태입니다."
                action  = "푸시/SMS 리마인드와 재방문 쿠폰으로 재활성화 캠페인을 진행합니다."
            else:
                problem = "최근 활동이 정상 고객보다 활발합니다."
                action  = "기존 혜택 수준을 유지하고 과도한 프로모션은 피합니다."

        elif feat == "PurchaseFrequency":
            if zval < 0:
                problem = "구매 간격이 길어져 재구매 속도가 느려진 상태입니다."
                action  = "정기구독/번들/바우처로 다음 구매 시점을 앞당기도록 제안합니다."
            else:
                problem = "구매 빈도가 정상 고객보다 높은 편입니다."
                action  = "우수 고객 케어(추가 혜택, VIP 편입 대상)로 관리합니다."

        elif feat == "AverageSatisfactionScore":
            if zval < 0:
                problem = "만족도가 정상 고객보다 낮습니다."
                action  = "케어 콜 및 보상 제안으로 불만 요소를 확인하고 개선합니다."
            else:
                problem = "만족도가 정상 고객보다 높습니다."
                action  = "긍정 후기를 유도하고, 충성 고객 프로그램으로 유입을 확대합니다."

        elif feat == "NegativeExperienceIndex":
            if up:
                problem = "불만/클레임 관련 신호가 정상 고객보다 많습니다."
                action  = "주요 클레임 유형을 정리해 근본 원인을 제거하고, 관련 티켓을 우선 처리합니다."
            else:
                problem = "부정 경험 지수는 낮지만, 개별 이력 확인은 필요합니다."
                action  = "최근 이슈가 없는지 모니터링 수준으로 관리합니다."

        elif feat == "EmailEngagementRate":
            if zval < 0:
                problem = "이메일을 거의 열어보지 않아 채널 효과가 떨어집니다."
                action  = "앱 푸시/SMS 등 다른 채널로 전환하고, 제목·발신자 A/B 테스트를 진행합니다."
            else:
                problem = "이메일 참여율이 높아 커뮤니케이션 채널로 효과적입니다."
                action  = "중요 공지·프로모션을 이메일 중심으로 설계합니다."

        elif feat == "TotalEngagementScore":
            if zval < 0:
                problem = "앱/웹 전체 활동 수준이 눈에 띄게 낮아진 상태입니다."
                action  = "온보딩/재온보딩 캠페인으로 핵심 기능을 다시 안내하고 알림 설정을 유도합니다."
            else:
                problem = "앱/웹 활동이 높아 충성 고객에 가깝습니다."
                action  = "추가 혜택보다 경험 품질(속도, 오류)을 우선 관리합니다."

        return problem, action

    # 1) 상단 카드: 위험 방향으로 많이 벗어난 요인 Top3
    driver_items = list(drivers.items())
    bad_items = [(f, float(zv)) for f, zv in driver_items if is_bad(f, float(zv))]
    if not bad_items:   # 모두 애매하면 그냥 상위 3개 사용
        bad_items = [(f, float(zv)) for f, zv in driver_items]
    top3 = bad_items[:3]

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    for i, (feat, zval) in enumerate(top3):
        if i >= 3:
            break
        col = cols[i]
        with col:
            badge, sev_text = severity_badge_and_text(zval)
            problem, action = describe_problem_action(feat, zval)

            st.markdown(
                f"**{NAME.get(feat, feat)}**  \n"
                f"{badge} · 정상 고객과 차이가 **{sev_text}**입니다.\n\n"
                f"- **문제 요약**: {problem}\n"
                f"- **권장 액션**: {action}"
            )

    # 2) 상세 테이블(Top 5) – 숫자 z점수 대신 텍스트로 영향 정도만 표시
    rows_drv = []
    for feat, zval in list(drivers.items())[:5]:
        zval = float(zval)
        _, sev_text = severity_badge_and_text(zval)
        _, action = describe_problem_action(feat, zval)
        rows_drv.append({
            "요인": NAME.get(feat, feat),
            "현재": float(row[feat]),
            "정상군 평균": float(mu[feat]),
            "영향 정도": sev_text,
            "권장 액션": action,
        })
    drv_view = pd.DataFrame(rows_drv)

    def style_severity(series: pd.Series):
        if series.name != "영향 정도":
            return [""] * len(series)
        styles = []
        for v in series:
            if v == "매우 큼":
                a = 0.90
            elif v == "큼":
                a = 0.65
            elif v == "보통":
                a = 0.40
            else:  # 작음
                a = 0.15
            styles.append(f"background-color: rgba(255,0,0,{a:.2f})")
        return styles

    styler_drv = (
        drv_view.style
        .format({"현재": "{:.2f}", "정상군 평균": "{:.2f}"})
        .hide(axis="index")
        .apply(style_severity, axis=0)
    )
    st.markdown(styler_drv.to_html(escape=False), unsafe_allow_html=True)

else:
    st.info("드라이버 분석을 위한 컬럼/정상군 기준이 부족합니다.")

# -------------------------------
# 📬 맞춤 문자 생성 / 발송 (개선본)
# -------------------------------
st.markdown("---")
st.subheader("📨 맞춤 문자 생성/발송")

# ----- 이하 SMS 부분은 동일 -----
import math

def sms_segments_korean(text: str):
    n = len(text or "")
    if n <= 70:
        return 1, 70 - n, n
    else:
        seg = 1 + math.ceil((n - 70) / 67.0)
        remain = (67 - ((n - 70) % 67)) % 67
        return seg, remain, n

def limit_for_segments(target_segments: int) -> int:
    if target_segments <= 1:
        return 70
    return 70 + 67 * (target_segments - 1)

def top_risk_reasons_natural(drivers_series: pd.Series) -> list[str]:
    if drivers_series is None or drivers_series.empty:
        return []
    feat, z = sorted(
        drivers_series.items(),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )[0]
    if feat == "CSFrequency":
        sent = "최근 상담이 자주 발생해 많이 번거로우셨을 수 있습니다."
    elif feat == "NegativeExperienceIndex":
        sent = "이용 과정에서 불편이나 클레임이 있었던 것으로 보입니다."
    elif feat == "AverageSatisfactionScore":
        sent = "만족도 응답에서 기대에 못 미친 부분이 있었습니다."
    elif feat in ["RecencyProxy", "AvgPurchaseInterval", "PurchaseFrequency"]:
        sent = "최근 이용·구매 빈도가 줄어든 상태입니다."
    elif feat in ["EmailEngagementRate", "TotalEngagementScore"]:
        sent = "앱·이메일 활동이 예전보다 줄어든 상태입니다."
    else:
        sent = "이용 패턴에 변동이 있는 고객으로 분석되었습니다."
    return [sent]

def detect_message_theme(drivers_series: pd.Series) -> str:
    if drivers_series is None or drivers_series.empty:
        return "promo"
    z = drivers_series.to_dict()
    def gt(name, thr=0.8):
        return abs(float(z.get(name, 0))) >= thr and float(z.get(name, 0)) > 0
    def lt(name, thr=0.8):
        return abs(float(z.get(name, 0))) >= thr and float(z.get(name, 0)) < 0
    if gt("CSFrequency") or gt("NegativeExperienceIndex") or lt("AverageSatisfactionScore"):
        return "care"
    if gt("RecencyProxy") or gt("AvgPurchaseInterval") or lt("PurchaseFrequency"):
        return "winback"
    if lt("EmailEngagementRate") or lt("TotalEngagementScore"):
        return "engage"
    return "promo"

def compose_variants(
    theme: str,
    tone: str,
    customer_id: str,
    brand: str,
    benefit: str,
    expiry: "datetime.date|str",
    landing_url: str,
    cs_contact: str,
    optout: str,
    target_segments: int,
    reason_sentence: str | None = None,
):
    exp_str = ""
    try:
        import datetime as _dt
        if isinstance(expiry, _dt.date):
            exp_str = expiry.strftime("%Y-%m-%d")
        elif expiry:
            exp_str = str(expiry)
    except Exception:
        if expiry:
            exp_str = str(expiry)

    if tone == "친근":
        hi_short = f"[{brand}] {customer_id}님"
        hi_long  = f"[{brand}] {customer_id}님 안녕하세요."
    elif tone == "긴급(한정)":
        hi_short = f"[{brand}] {customer_id} 고객님"
        hi_long  = f"[{brand}] {customer_id} 고객님, 중요한 안내드립니다."
    else:
        hi_short = f"[{brand}] {customer_id} 고객님"
        hi_long  = f"[{brand}] {customer_id} 고객님 안녕하세요."

    use_reason = bool(reason_sentence and target_segments > 1)
    rs = (reason_sentence or "").rstrip()

    if target_segments == 1:
        if theme == "care":
            A = f"{hi_short}, 이용 중 불편을 드려 죄송합니다. 사과의 마음으로 {benefit}을 드립니다."
            B = f"{hi_short}, 서비스 이용에 불편이 있으셨다면 죄송합니다. 보상으로 {benefit}을 준비했습니다."
        elif theme == "winback":
            A = f"{hi_short}, 오랜만에 인사드립니다. 다시 방문 시 {benefit}을 드립니다."
            B = f"{hi_short}, 최근 이용이 줄어 아쉬운 마음에 {benefit}을 준비했습니다."
        elif theme == "engage":
            A = f"{hi_short}, 새 혜택과 이벤트가 열렸습니다. {benefit}을 확인해 주세요."
            B = f"{hi_short}, 혜택을 놓치지 않도록 {benefit}을 안내드립니다."
        else:
            A = f"{hi_short}께 {benefit} 혜택을 준비했습니다."
            B = f"{hi_short}, 지금 {benefit}을 이용하실 수 있습니다."
    else:
        if theme == "care":
            mid = (rs + " ") if use_reason else ""
            A = (
                f"{hi_long} 이용 중 불편을 드려 진심으로 죄송합니다. "
                f"{mid}사과의 마음으로 {benefit}을 준비했으며 {exp_str}까지 사용 가능합니다."
            )
            B = (
                f"{hi_long} 서비스 이용 과정에서 만족스럽지 못하셨던 부분이 있었던 것 같습니다. "
                f"{mid}{benefit}을 제공해 드리니, 다시 한번 편하게 이용해 주시면 감사하겠습니다."
            )
        elif theme == "winback":
            mid = (rs + " ") if use_reason else ""
            A = (
                f"{hi_long} 요즘 자주 뵙지 못해 먼저 연락드립니다. "
                f"{mid}다시 방문해 주시는 고객님께 감사의 뜻으로 {benefit}을 드립니다. "
                f"{exp_str}까지 사용 가능합니다."
            )
            B = (
                f"{hi_long} 한동안 이용이 뜸하셔서 아쉬운 마음에 연락드립니다. "
                f"{mid}준비된 {benefit}으로 다시 한번 혜택을 경험해 보시면 좋겠습니다."
            )
        elif theme == "engage":
            mid = (rs + " ") if use_reason else ""
            A = (
                f"{hi_long} 새 혜택과 이벤트가 업데이트되었습니다. "
                f"{mid}고객님께 맞는 {benefit}을 마련해 두었으니 {exp_str} 전에 한 번 확인해 주세요."
            )
            B = (
                f"{hi_long} 혜택과 알림을 더 알차게 이용하실 수 있도록 {benefit}을 추가했습니다. "
                f"{mid}간단히 접속만으로 적용되니 놓치지 마세요. {exp_str}까지입니다."
            )
        else:
            A = (
                f"{hi_long} 고객님께 어울리는 {benefit}을 준비했습니다. "
                f"{exp_str}까지 사용 가능하니 쇼핑에 참고 부탁드립니다."
            )
            B = (
                f"{hi_long} 현재 고객님께 제공되는 {benefit}이 오픈되었습니다. "
                f"{exp_str} 전까지 자유롭게 사용해 보시고, 부족한 점이 있다면 언제든 알려 주세요."
            )

    url = f" 바로가기: {landing_url}" if landing_url else ""
    cs  = f" 문의: {cs_contact}" if cs_contact else ""
    oo  = f" {optout}" if optout else ""

    vA = (A + url + cs + oo).strip()
    vB = (B + url + cs + oo).strip()
    return [vA, vB]

def fit_to_target(text: str, target_segments: int) -> str:
    limit = limit_for_segments(target_segments)
    if len(text) <= limit:
        return text
    t = text
    for key in [" 수신거부", " 문의:", " 바로가기:"]:
        if len(t) <= limit:
            break
        idx = t.rfind(key)
        if idx != -1:
            t = t[:idx].strip()
    if len(t) <= limit:
        return t
    sentences = []
    buf = ""
    for ch in t:
        buf += ch
        if ch in ".!?" or buf.endswith("요.") or buf.endswith("다.") or buf.endswith("니다."):
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    if sentences:
        result = ""
        for s in sentences:
            if len((result + " " + s).strip()) <= limit:
                result = (result + " " + s).strip()
            else:
                break
        if result:
            t = result
    if len(t) <= limit:
        return t
    t = t[:limit]
    last_space = t.rfind(" ")
    if last_space > 0 and last_space > limit * 0.5:
        t = t[:last_space]
    return t.rstrip(" ,.;") + "…"

col_left, col_right = st.columns([1, 1])
with col_left:
    brand = st.text_input("브랜드명", value="브랜드")
    benefit = st.text_input("혜택(예: 5% 할인, 무료배송, 1만원 쿠폰)", value="5% 할인 쿠폰")
    expiry = st.date_input("혜택 만료일", value=datetime.date.today() + datetime.timedelta(days=7))
    tone = st.selectbox("톤/스타일", ["정중", "친근", "긴급(한정)"], index=0)
with col_right:
    theme_choice = st.selectbox("메시지 타입", ["자동 추천", "사과/케어", "휴면/재활성", "참여 활성화", "일반 프로모션"], index=0)
    landing_url = st.text_input("랜딩 URL(선택)", value="")
    cs_contact = st.text_input("문의 채널(선택, 예: 080-000-0000 / 챗봇 링크)", value="")
    optout = st.text_input("수신거부 문구(선택)", value="수신거부: 수신중지")
    to_phone = st.text_input("수신번호(To, 선택: Twilio 발송 시 사용)", value="", placeholder="+8210XXXXYYYY")

target_segments = st.radio("목표 길이", ["1건(≤70자)", "2건(≤137자)"], index=0, horizontal=True)
target_segments = 1 if target_segments.startswith("1") else 2

reasons_natural = top_risk_reasons_natural(drivers) if "drivers" in locals() and drivers is not None else []
reason_for_sms = reasons_natural[0] if (reasons_natural and target_segments > 1) else None

if theme_choice == "자동 추천":
    theme = detect_message_theme(drivers) if "drivers" in locals() and drivers is not None else "promo"
else:
    theme = {
        "사과/케어": "care",
        "휴면/재활성": "winback",
        "참여 활성화": "engage",
        "일반 프로모션": "promo",
    }[theme_choice]

variants = compose_variants(
    theme=theme,
    tone=tone,
    customer_id=customer_id,
    brand=brand,
    benefit=benefit,
    expiry=expiry,
    landing_url=landing_url,
    cs_contact=cs_contact,
    optout=optout,
    target_segments=target_segments,
    reason_sentence=reason_for_sms,
)

vA, vB = variants[0], variants[1]
best = fit_to_target(vA, target_segments)
alt  = fit_to_target(vB, target_segments)
final_msg = best if len(best) <= len(alt) else alt

msg = st.text_area("문자 내용(편집 가능)", value=final_msg, height=140)
seg, remain, nchar = sms_segments_korean(msg)
st.caption(f"{nchar}자 · 추정 {seg}건(현재 세그먼트 남은 {remain}자)  *UCS-2 기준 70/67 규칙*")

with st.expander("추천안 A/B 미리보기", expanded=False):
    st.markdown("**A 후보**")
    st.code(best, language="text")
    st.markdown("**B 후보**")
    st.code(alt, language="text")

st.download_button(
    "⬇️ TXT로 저장",
    data=msg.encode("utf-8"),
    file_name=f"{customer_id}_sms.txt",
    mime="text/plain",
)

# -------------------------------
# Twilio 발송
# -------------------------------
with st.expander("☁️ Twilio 설정(선택: 설정 시 실제 발송)", expanded=False):
    st.caption("설정 후 아래 버튼으로 실제 문자 발송이 가능합니다. 미설정 시 '문자 생성/복사'만 사용하세요.")
    import os
    def _get_secret(name, default=""):
        try:
            return st.secrets.get(name, os.getenv(name, default))
        except Exception:
            return os.getenv(name, default)

    default_sid  = _get_secret("TWILIO_ACCOUNT_SID")
    default_tok  = _get_secret("TWILIO_AUTH_TOKEN")
    default_from = _get_secret("TWILIO_FROM", "+15005550006")
    default_msid = _get_secret("TWILIO_MESSAGING_SERVICE_SID", "")

    tw_sid   = st.text_input("Twilio Account SID", value=default_sid, type="password")
    tw_token = st.text_input("Twilio Auth Token", value=default_tok, type="password")
    from_phone = st.text_input("발신번호(From, E.164)", value=default_from, help="Twilio 콘솔에서 보유한 SMS 가능 번호")
    msid = st.text_input("Messaging Service SID (선택)", value=default_msid, help="값이 있으면 From 대신 Messaging Service 사용")

    if st.button("📤 문자 발송 (Twilio)", type="primary"):
        if not (tw_sid and tw_token and msg and to_phone and (msid or from_phone)):
            st.error("SID/Token/발신(or MSID)/수신번호/메시지를 모두 입력하세요.")
        else:
            try:
                from twilio.rest import Client  # type: ignore
                tw = Client(tw_sid, tw_token)
                if msid:
                    m = tw.messages.create(body=msg, to=to_phone, messaging_service_sid=msid)
                else:
                    m = tw.messages.create(body=msg, from_=from_phone, to=to_phone)
                st.success(f"발송 완료! SID: {m.sid}")
            except ImportError:
                st.error("twilio 패키지가 없습니다. 아래 명령으로 설치하세요:\n\n`python -m pip install twilio`")
            except Exception as e:
                st.error(f"발송 실패: {e}")

st.caption("※ 반드시 수신 동의/옵트아웃 등 관련 법규를 준수하여 발송하세요.")

# -------------------------------
# 액션 로그
# -------------------------------
st.markdown("---")
st.subheader("📌 액션 기록 / 히스토리")

@st.cache_resource(show_spinner=False)
def get_conn():
    conn = sqlite3.connect("actions.db", check_same_thread=False)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id TEXT,
        action TEXT,
        note TEXT,
        ts TEXT,
        owner TEXT,
        status TEXT
    )
    """)
    return conn

conn = get_conn()
c1, c2, c3 = st.columns(3)
with c1:
    action = st.selectbox("액션", ["콜백","쿠폰 발송","VIP 케어","이메일 발송","재참여 캠페인","SMS"])
with c2:
    owner = st.text_input("담당자", value="")
with c3:
    status = st.selectbox("상태", ["open","done","hold"])
note = st.text_area("메모")
if st.button("저장"):
    conn.execute(
        "INSERT INTO actions (customer_id, action, note, ts, owner, status) VALUES (?,?,?,?,?,?)",
        (customer_id, action, note, datetime.datetime.utcnow().isoformat(), owner, status)
    )
    conn.commit()
    st.success("저장되었습니다.")

hist = pd.read_sql_query(
    "SELECT ts, action, owner, status, note FROM actions WHERE customer_id = ? ORDER BY ts DESC",
    conn, params=(customer_id,)
)

# 히스토리 컬럼 한글 라벨링
hist_display = hist.rename(columns={
    "ts": "기록일시",
    "action": "액션",
    "owner": "담당자",
    "status": "상태",
    "note": "메모",
})
st.dataframe(hist_display, use_container_width=True)

# 하단 네비
st.markdown("---")
try:
    st.page_link("app_enhanced.py", label="← 대시보드로 돌아가기", icon="🏠")
except Exception:
    st.markdown("[← 대시보드로 돌아가기](/)")