# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, datetime, json, os, math
from urllib.parse import unquote

st.set_page_config(page_title="👤 고객 상세", layout="wide")

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
        try: code_map.update({int(k): v for k, v in code_json.items()})
        except Exception: pass
    label_json = _load_json(GENDER_LABEL_MAP_PATH)
    if label_json:
        try: code_map.update({int(k): v for k, v in label_json.items()})
        except Exception: pass
    if "gender_label_map" in st.session_state and isinstance(st.session_state["gender_label_map"], dict):
        try: code_map.update({int(k): v for k, v in st.session_state["gender_label_map"].items()})
        except Exception: pass

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
# Header (상단: 대시보드로 버튼, 그 아래 좌측 아이콘 타이틀)
# -------------------------------
try:
    st.page_link("app_enhanced.py", label="← 대시보드로", icon="🏠")
except Exception:
    st.markdown("[← 대시보드로](/)")

st.title("👤 고객 상세")
st.caption(f"CustomerID: {customer_id}")

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

    base_pairs = [("CustomerID", customer_id)]
    if sex_label is not None:
        base_pairs.append(("성별", sex_label))
    if age_int is not None:
        base_pairs.append(("나이", age_int))
    for c, label in [
        ("IncomeLevel", "소득수준"),
        ("CustomerTenure", "이용 개월(추정)"),
        ("RepeatCustomer", "리피트 고객 코드"),
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
    feat_pairs = [(c, row[c]) for c in feat_cols]
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
        st.caption(f"Raw={val:.2f}, p99={scale:.2f}")
    else:
        st.info("ChurnRiskScore 컬럼이 없어 게이지를 표시할 수 없습니다.")

with g2:
    st.subheader("상태")
    if all(exists(c) for c in ["Both_ChurnFlag","IF_ChurnFlag","AE_ChurnFlag"]):
        if int(row["Both_ChurnFlag"]) == 1:
            st.error("고신뢰 이탈 (IF & AE)")
        elif int(row["IF_ChurnFlag"]) == 1:
            st.warning("불만형 이탈 신호 (IF)")
        elif int(row["AE_ChurnFlag"]) == 1:
            st.info("조용한 이탈 신호 (AE)")
        else:
            st.success("정상")
    else:
        st.caption("플래그 컬럼 없음")

# -------------------------------
# 활동/만족 지표 - 전체 대비 분위 & 리스크 시각화
# -------------------------------
st.markdown("---")
st.subheader("📊 전체 대비 위치 & 리스크(%)")

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
        "지표": c,
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
    "PurchaseFrequency","CSFrequency","RecencyProxy",
    "AverageSatisfactionScore","NegativeExperienceIndex",
    "EmailEngagementRate","TotalEngagementScore"
]
driver_cols = [c for c in candidate_cols if exists(c)]

drivers = None
if driver_cols and exists("Both_ChurnFlag"):
    healthy = df[df["Both_ChurnFlag"] == 0][driver_cols].copy()
    mu = healthy.mean(numeric_only=True)
    sigma = healthy.std(numeric_only=True).replace(0, 1e-6)
    z = ((row[driver_cols] - mu) / sigma).astype(float)
    drivers = z.sort_values(key=lambda s: s.abs(), ascending=False)

    # Top 3 카드
    top3 = list(drivers.items())[:3]
    c1, c2, c3 = st.columns(3)
    NAME = {
        "PurchaseFrequency":"구매 빈도", "CSFrequency":"상담 빈도", "RecencyProxy":"활동저하 지수",
        "AverageSatisfactionScore":"만족도", "NegativeExperienceIndex":"부정경험 지수",
        "EmailEngagementRate":"이메일 참여율", "TotalEngagementScore":"총 참여 점수",
    }
    def recommend(feat: str, zval: float):
        if feat == "CSFrequency": return "이슈 가능성↑ → 시니어 상담 배정, 불만 원인 즉시 해결"
        if feat == "RecencyProxy": return "휴면 징후↑ → 리엑티베이션(푸시/SMS), 재방문 쿠폰"
        if feat == "AverageSatisfactionScore": return "만족도↓ → 케어 콜, 품질/배송 개선, 보상 제공"
        if feat == "NegativeExperienceIndex": return "부정경험↑ → 근본 원인 제거, 티켓 즉시 처리"
        if feat == "EmailEngagementRate": return "참여율↓ → 채널 전환(앱푸시/SMS), 제목/발신자 A/B"
        if feat == "TotalEngagementScore": return "참여점수↓ → (재)온보딩, 알림 설정 유도"
        if feat == "PurchaseFrequency": return "구매빈도↓ → 바우처/정기구독/번들로 간격 단축"
        return "개인화 혜택과 빠른 CS 응대"

    for i, (feat, zval) in enumerate(top3):
        with (c1 if i==0 else c2 if i==1 else c3):
            sev = abs(float(zval))
            sev_badge = "🔴 높음" if sev >= 2.0 else ("🟠 중간" if sev >= 1.0 else "🟡 낮음")
            st.markdown(f"**{NAME.get(feat, feat)}**  \n*z={float(zval):+.2f}*  \n{sev_badge}")
            st.write(f"→ **{recommend(feat, float(zval))}**")

    # 상세 테이블(Top 5)
    rows_drv = []
    for feat, zval in list(drivers.items())[:5]:
        rows_drv.append({
            "요인": NAME.get(feat, feat),
            "현재": float(row[feat]),
            "정상군 평균": float(mu[feat]),
            "편차(z)": float(zval),
            "권장 액션": recommend(feat, float(zval)),
        })
    drv_view = pd.DataFrame(rows_drv)

    def style_z(series: pd.Series):
        if series.name != "편차(z)":
            return [""] * len(series)
        vals = series.abs()
        vmax = max(vals.max(), 1.0)
        styles = []
        for v in vals:
            a = 0.15 + 0.75 * (float(v) / vmax)
            a = max(0, min(1, a))
            styles.append(f"background-color: rgba(255,0,0,{a:.2f})")
        return styles

    styler_drv = drv_view.style.format({
        "현재":"{:.2f}", "정상군 평균":"{:.2f}", "편차(z)":"{:+.2f}"
    }).hide(axis="index").apply(style_z, axis=0)
    st.markdown(styler_drv.to_html(escape=False), unsafe_allow_html=True)
else:
    st.info("드라이버 분석을 위한 컬럼/정상군 기준이 부족합니다.")

# -------------------------------
# 📬 맞춤 문자 생성 / 발송 (향상본)
# -------------------------------
st.markdown("---")
st.subheader("📨 맞춤 문자 생성/발송")

import math

# ──────────────────────
# 0) 유틸: 문자 길이/세그먼트 계산 (한글 UCS-2: 70/67 규칙)
# ──────────────────────
def sms_segments_korean(text: str):
    n = len(text or "")
    if n <= 70:
        return 1, 70 - n, n
    else:
        seg = 1 + math.ceil((n - 70) / 67.0)
        remain = (67 - ((n - 70) % 67)) % 67
        return seg, remain, n

def limit_for_segments(target_segments: int) -> int:
    # 1건: 70자, 2건: 70+67 = 137자
    if target_segments <= 1:
        return 70
    return 70 + 67 * (target_segments - 1)

# ──────────────────────
# 1) 위험 사유(리스크 드라이버) → 자연어 문구
# ──────────────────────
def top_risk_reasons_natural(drivers_series: pd.Series) -> list[str]:
    if drivers_series is None or drivers_series.empty:
        return []
    # 리스크 방향 정의
    dir_map = {
        "CSFrequency": "higher_worse",
        "RecencyProxy": "higher_worse",
        "NegativeExperienceIndex": "higher_worse",
        "AvgPurchaseInterval": "higher_worse",
        "PurchaseFrequency": "lower_worse",
        "AverageSatisfactionScore": "lower_worse",
        "EmailEngagementRate": "lower_worse",
        "TotalEngagementScore": "lower_worse",
    }
    # 후보(나쁜 방향만 우선)
    cand = []
    for feat, zval in drivers_series.items():
        d = dir_map.get(feat, "neutral")
        bad = (zval > 0 and d == "higher_worse") or (zval < 0 and d == "lower_worse")
        if d == "neutral":
            bad = abs(zval) >= 1.0
        if bad:
            cand.append((feat, float(zval)))
    if not cand:
        cand = [(k, float(v)) for k, v in drivers_series.items()]
    cand = sorted(cand, key=lambda x: abs(x[1]), reverse=True)[:2]

    reasons = []
    for feat, z in cand:
        if feat == "RecencyProxy":
            reasons.append("최근 이용이 줄어든 것으로 보여")
        elif feat == "PurchaseFrequency":
            reasons.append("구매 간격이 길어져")
        elif feat == "CSFrequency":
            reasons.append("상담 이력이 잦아 불편하셨을 수 있어")
        elif feat == "AverageSatisfactionScore":
            reasons.append("만족도가 낮게 확인되어")
        elif feat == "NegativeExperienceIndex":
            reasons.append("부정 경험 신호가 확인되어")
        elif feat == "EmailEngagementRate":
            reasons.append("이메일 확인이 어려우신 것 같아")
        elif feat == "TotalEngagementScore":
            reasons.append("앱/웹 활동이 줄어")
        elif feat == "AvgPurchaseInterval":
            reasons.append("구매 간격이 늘어나")
        else:
            reasons.append("이용 패턴에 변동이 있어")
    return reasons[:2]

# ──────────────────────
# 2) 드라이버 기반 메시지 타입 자동 판정
#     care(사과/케어) / winback(휴면/재활성) / engage(참여 활성화) / promo(일반)
# ──────────────────────
def detect_message_theme(drivers_series: pd.Series) -> str:
    if drivers_series is None or drivers_series.empty:
        return "promo"
    # z-score 기준으로 주요 신호 파악
    z = drivers_series.to_dict()
    def gt(name, thr=0.8):   # 높을수록 나쁜 지표
        return abs(float(z.get(name, 0))) >= thr and float(z.get(name, 0)) > 0
    def lt(name, thr=0.8):   # 낮을수록 나쁜 지표
        return abs(float(z.get(name, 0))) >= thr and float(z.get(name, 0)) < 0

    if gt("CSFrequency") or gt("NegativeExperienceIndex") or lt("AverageSatisfactionScore"):
        return "care"      # 불만/사과형
    if gt("RecencyProxy") or gt("AvgPurchaseInterval") or lt("PurchaseFrequency"):
        return "winback"   # 휴면/재활성
    if lt("EmailEngagementRate") or lt("TotalEngagementScore"):
        return "engage"    # 참여 활성화
    return "promo"

# ──────────────────────
# 3) 톤 & 타입별 템플릿 (A/B) + 길이 자동 맞춤
# ──────────────────────
def compose_variants(theme: str, tone: str, customer_id: str, brand: str, benefit: str,
                     expiry: "datetime.date|str", landing_url: str, cs_contact: str, optout: str):
    exp_str = ""
    try:
        import datetime as _dt
        if isinstance(expiry, _dt.date):
            exp_str = f"만료 {expiry.strftime('%Y-%m-%d')}"
        elif expiry:
            exp_str = f"만료 {expiry}"
    except Exception:
        if expiry: exp_str = f"만료 {expiry}"

    # 인사말(톤)
    if tone == "친근":
        hi = f"[{brand}] {customer_id}님,"
    elif tone == "긴급(한정)":
        hi = f"[{brand}] {customer_id} 고객님,"
    else:  # 정중
        hi = f"[{brand}] {customer_id} 고객님,"

    # 타입별 바디 A/B
    if theme == "care":
        A = f"{hi} 불편 드려 죄송합니다. 사과의 뜻으로 {benefit} 드립니다. {exp_str}."
        B = f"{hi} 이용 중 불편을 확인했습니다. {benefit} 제공드립니다. {exp_str}."
    elif theme == "winback":
        A = f"{hi} 오랜만이에요. 돌아오실 수 있게 {benefit} 준비했어요. {exp_str}."
        B = f"{hi} 최근 이용이 적어 아쉬워요. 지금 {benefit}로 다시 만나세요. {exp_str}."
    elif theme == "engage":
        A = f"{hi} 새 혜택을 놓치지 마세요. 맞춤 {benefit} 드립니다. {exp_str}."
        B = f"{hi} 참여 혜택을 강화했어요. 전용 {benefit} 확인해 주세요. {exp_str}."
    else:  # promo
        A = f"{hi} 고객님께 맞춘 {benefit} 안내드립니다. {exp_str}."
        B = f"{hi} 지금 적용 가능한 {benefit}가 준비됐습니다. {exp_str}."

    # CTA/추가
    url = f" 바로가기: {landing_url}" if landing_url else ""
    cs  = f" 문의: {cs_contact}" if cs_contact else ""
    oo  = f" {optout}" if optout else ""

    # 두 가지 후보
    vA = A + url + cs + oo
    vB = B + url + cs + oo
    return [vA.strip(), vB.strip()]

def fit_to_target(text: str, target_segments: int) -> str:
    # 길이 초과 시 제거 우선순위: 수신거부 → 문의 → URL → 만료문구 일부 → 인사말 축약
    limit = limit_for_segments(target_segments)
    if len(text) <= limit:
        return text

    # 단계적 축소
    t = text
    # 1) 수신거부 제거
    if "수신거부" in t and len(t) > limit:
        i = t.rfind("수신거부")
        if i > -1: t = t[:i].strip()
    # 2) 문의 제거
    if " 문의:" in t and len(t) > limit:
        i = t.rfind(" 문의:")
        if i > -1: t = t[:i].strip()
    # 3) URL 제거
    if "바로가기:" in t and len(t) > limit:
        i = t.rfind("바로가기:")
        if i > -1: t = t[:i].strip()
    # 4) 만료문구 줄이기: "만료 YYYY-MM-DD" → "만료 YYYYMMDD"
    t = t.replace("만료 ", "만료")
    import re
    t = re.sub(r"만료(\s*)?(\d{4})-(\d{2})-(\d{2})", r"만료\2\3\4", t)
    # 5) 인사말 축약: "고객님," → "님,"
    if len(t) > limit:
        t = t.replace(" 고객님,", " 님,")
        t = t.replace(" 고객님,", " 님,")
    # 마지막 방어: 초과면 절단(말줄임)
    return t[:limit]

# ──────────────────────
# 4) 입력 파라미터 UI
# ──────────────────────
col_left, col_right = st.columns([1,1])
with col_left:
    brand = st.text_input("브랜드명", value="브랜드")
    benefit = st.text_input("혜택(예: 5% 할인, 무료배송, 1만원 쿠폰)", value="5% 할인 쿠폰")
    expiry = st.date_input("혜택 만료일", value=datetime.date.today() + datetime.timedelta(days=7))
    tone = st.selectbox("톤/스타일", ["정중", "친근", "긴급(한정)"], index=0)
with col_right:
    theme_choice = st.selectbox("메시지 타입", ["자동 추천","사과/케어","휴면/재활성","참여 활성화","일반 프로모션"], index=0)
    landing_url = st.text_input("랜딩 URL(선택)", value="")
    cs_contact = st.text_input("문의 채널(선택, 예: 080-000-0000 / 챗봇 링크)", value="")
    optout = st.text_input("수신거부 문구(선택)", value="수신거부: 수신중지")
    to_phone = st.text_input("수신번호(To, 선택: Twilio 발송 시 사용)", value="", placeholder="+8210XXXXYYYY")
target_segments = st.radio("목표 길이", ["1건(≤70자)", "2건(≤137자)"], index=0, horizontal=True)
target_segments = 1 if target_segments.startswith("1") else 2

# ──────────────────────
# 5) 드라이버 기반 사유 & 타입 선택 → 후보 2개 생성 → 길이에 맞춰 자동 조정
# ──────────────────────
reasons_natural = top_risk_reasons_natural(drivers) if 'drivers' in locals() and drivers is not None else []
# 타입 자동 판정
if theme_choice == "자동 추천":
    theme = detect_message_theme(drivers) if 'drivers' in locals() and drivers is not None else "promo"
else:
    theme = {"사과/케어":"care","휴면/재활성":"winback","참여 활성화":"engage","일반 프로모션":"promo"}[theme_choice]

# 후보 2개 생성(A/B)
variants = compose_variants(theme, tone, customer_id, brand, benefit, expiry, landing_url, cs_contact, optout)

# 사유 문구를 한 줄 덧붙이되, 길이 내에서만 추가
reason_line = ""
if reasons_natural:
    # 가장 중요한 사유 1개만 짧게
    reason_line = f" ({reasons_natural[0]})"
vA = variants[0] + reason_line
vB = variants[1] + reason_line

# 길이에 맞게 자동 축소
best = fit_to_target(vA, target_segments)
alt  = fit_to_target(vB, target_segments)

# 후보 선택 로직: 더 짧은 쪽 우선
final_msg = best if len(best) <= len(alt) else alt

# ──────────────────────
# 6) 편집/미리보기/다운로드
# ──────────────────────
msg = st.text_area("문자 내용(편집 가능)", value=final_msg, height=140)
seg, remain, nchar = sms_segments_korean(msg)
st.caption(f"{nchar}자 · 추정 {seg}건(현재 세그먼트 남은 {remain}자)  *UCS-2 기준 70/67 규칙*")

# 추천안 A/B 미리보기
with st.expander("추천안 A/B 미리보기", expanded=False):
    st.markdown("**A**")
    st.code(best, language="text")
    st.markdown("**B**")
    st.code(alt,  language="text")

st.download_button("⬇️ TXT로 저장", data=msg.encode("utf-8"), file_name=f"{customer_id}_sms.txt", mime="text/plain")

# ──────────────────────
# 7) (선택) Twilio 발송
# ──────────────────────
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
st.dataframe(hist, use_container_width=True)

# 하단 네비
st.markdown("---")
try:
    st.page_link("app_enhanced.py", label="← 대시보드로 돌아가기", icon="🏠")
except Exception:
    st.markdown("[← 대시보드로 돌아가기](/)")