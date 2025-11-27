# pages/03_VIP_Insights.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os, json
from urllib.parse import quote

# 외부 유틸: 프로젝트 루트의 utils_vip.py
from utils_vip import (
    compute_vip_propensity_score,
    select_vip_candidates,
    backtest_metrics,
    roi_for_k,
)

# ---------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------
st.set_page_config(page_title="⭐ VIP 인사이트", layout="wide")

# ---------------------------------------------------------------------
# 표시 라벨(표시 전용)
# ---------------------------------------------------------------------
KOR_COL = {
    "CustomerID_clean": "고객ID",
    "GenderLabel": "성별",
    "CustomerLifetimeValue": "고객생애가치(CLV)",
    "PurchaseFrequency": "구매빈도",
    "AverageOrderValue": "평균주문금액(AOV)",
    "TotalPurchases": "총구매수",
    "AvgPurchaseInterval": "평균구매간격",
    "EmailEngagementRate": "이메일참여율",
    "MobileAppUsage": "앱사용",
    "TotalEngagementScore": "총참여점수",
    "AverageSatisfactionScore": "평균만족도",
    "NegativeExperienceIndex": "불편경험지수",
    "CSFrequency": "상담빈도",
    "Age": "나이",
    "AnnualIncome": "연소득",
    "Income": "연소득",
    "IF_AnomalyScore": "패턴이탈지수(IF)",
    "AE_ReconError": "정상패턴차이(AE)",
    "coverage": "데이터충분도",
}
def dlabel(c): return KOR_COL.get(c, c)
def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: dlabel(c) for c in df.columns})

# === 안전한 컬럼 선택 헬퍼(표 머리 한글화 이후 쓰기) ===
def to_display_cols(cols: list[str]) -> list[str]:
    return [KOR_COL.get(c, c) for c in cols]

def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    disp = to_display_cols(cols)
    return [c for c in disp if c in df.columns]

# ---------------------------------------------------------------------
# 성별 라벨 보장(대시보드와 동일)
# ---------------------------------------------------------------------
DEFAULT_CODE_TO_LABEL_KO = {1:"여성",3:"남성",5:"응답거부",4:"기타/미상",2:"남성",0:"여성"}
def _normalize_gender_text_to_label_ko(x)->str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return "미상"
    s = str(x).strip().lower()
    if s in {"m","male","man","남","남성"}: return "남성"
    if s in {"f","female","woman","여","여성"}: return "여성"
    if s in {"prefer not to say","decline to state","no answer"}: return "응답거부"
    if s in {"non-binary","nonbinary","genderqueer","agender","nb","other","기타"}: return "기타"
    return "기타"

def ensure_gender_label(df_hybrid: pd.DataFrame,
                        original_csv_path: str = "ecommerce_customer_data.csv",
                        code_map_path: str = "gender_code_map.json") -> pd.DataFrame:
    df = df_hybrid.copy()
    if os.path.exists(original_csv_path):
        try:
            raw = pd.read_csv(original_csv_path, usecols=["CustomerID","Gender"])
            raw["GenderLabel_from_raw"] = raw["Gender"].map(_normalize_gender_text_to_label_ko)
            df = df.merge(raw[["CustomerID","GenderLabel_from_raw"]], on="CustomerID", how="left")
        except Exception:
            df["GenderLabel_from_raw"] = np.nan
    else:
        df["GenderLabel_from_raw"] = np.nan
    code_map = DEFAULT_CODE_TO_LABEL_KO.copy()
    if os.path.exists(code_map_path):
        try:
            with open(code_map_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                code_map.update({int(k): v for k, v in loaded.items()})
        except Exception:
            pass
    label_from_code = df["Gender"].map(code_map) if "Gender" in df.columns else pd.Series(index=df.index, dtype="object")
    df["GenderLabel"] = df["GenderLabel_from_raw"].fillna(label_from_code)
    df.drop(columns=["GenderLabel_from_raw"], inplace=True)
    df["GenderLabel"] = df["GenderLabel"].fillna("미상")
    return df

# ---------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    base = pd.read_csv("ecommerce_customer_churn_hybrid_with_id.csv")
    if "CustomerID" in base.columns:
        def _clean_id(x):
            if pd.isna(x): return np.nan
            s = str(x).strip()
            return np.nan if (s=="" or s.lower() in {"nan","none","nat","null"}) else s
        base["CustomerID_clean"] = base["CustomerID"].map(_clean_id)
    base = ensure_gender_label(base)

    # 추가 피처 조인(있을 때만)
    if os.path.exists("ecommerce_customer_data_featured.csv"):
        feat = pd.read_csv("ecommerce_customer_data_featured.csv")
        if "CustomerID" in feat.columns:
            def _clean_id2(x):
                if pd.isna(x): return np.nan
                s = str(x).strip()
                return np.nan if (s=="" or s.lower() in {"nan","none","nat","null"}) else s
            feat["CustomerID_clean"] = feat["CustomerID"].map(_clean_id2)
            keep_cols = [c for c in feat.columns if c not in base.columns or c in
                         ["CustomerID","CustomerID_clean","CustomerLifetimeValue","AverageOrderValue",
                          "TotalPurchases","AvgPurchaseInterval","EmailEngagementRate","MobileAppUsage",
                          "TotalEngagementScore","AnnualIncome","Income"]]
            base = base.merge(feat[keep_cols], on=["CustomerID","CustomerID_clean"], how="left")
    return base

df = load_data()

# ---------------------------------------------------------------------
# 전역 필터(대시보드 공유) 적용
# ---------------------------------------------------------------------
sel_age = st.session_state.get("sel_age")
sel_gender_labels = st.session_state.get("sel_gender_labels", [])
premium_opt = st.session_state.get("premium_opt", "전체")

filtered = df.copy()
if sel_age and "Age" in filtered.columns:
    filtered = filtered[(filtered["Age"] >= sel_age[0]) & (filtered["Age"] <= sel_age[1])]
if sel_gender_labels and "GenderLabel" in filtered.columns:
    filtered = filtered[filtered["GenderLabel"].isin(sel_gender_labels)]
if "RepeatAndPremiumFlag" in filtered.columns and premium_opt != "전체":
    filtered = filtered[filtered["RepeatAndPremiumFlag"] == (1 if str(premium_opt).startswith("예") else 0)]

# ---------------------------------------------------------------------
# 상단 네비게이션
# ---------------------------------------------------------------------
try:
    st.page_link("app_enhanced.py", label="⬅️ 대시보드로", icon="🏠")
except Exception:
    st.markdown("[🏠 대시보드로](/)")

st.title("⭐ VIP 인사이트")
st.caption("VIP 정의와 전환 후보 선정을 한 화면에서 확인하고, 후보 리스트와 예상 ROI를 함께 확인합니다.")

# ---------------------------------------------------------------------
# 유틸(표 스타일/라벨)
# ---------------------------------------------------------------------
def qv(s: pd.Series, q: int|float) -> float|None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q/100.0)) if len(s)>0 else None

def table_css():
    st.markdown("""
    <style>
    #vip_table, #pot_table { width: 100% !important; table-layout: fixed; }
    #vip_table th, #vip_table td, #pot_table th, #pot_table td {
      padding: 10px 12px !important; line-height: 1.45; vertical-align: middle;
      white-space: normal !important; word-break: keep-all;
    }
    .badge { padding: 2px 6px; border-radius: 6px; font-size: 12px; line-height: 1;
             background: rgba(0,0,0,0.06); }
    .badge.gold { background: rgba(255,204,0,.18); border: 1px solid rgba(255,204,0,.35); }
    .badge.green{ background: rgba(52,199,89,.18); border: 1px solid rgba(52,199,89,.35); }
    .chip { display:inline-block; padding:2px 6px; background:rgba(0,0,0,.06); border-radius:6px; font-size:12px; }
    .barwrap { display:flex; align-items:center; gap:8px; }
    .bar    { flex:1; height:10px; background:rgba(0,0,0,0.08); border-radius:999px; overflow:hidden; }
    .bar .fill { height:100%; background: rgba(10,132,255,0.55); }
    </style>
    """, unsafe_allow_html=True)

def recommend_tags(row, ref_df):
    tags = []
    def add(t): tags.append(t)
    if "AverageOrderValue" in row and pd.notna(row["AverageOrderValue"]) and "AverageOrderValue" in ref_df.columns:
        thr = qv(ref_df["AverageOrderValue"], 85)
        if thr is not None and row["AverageOrderValue"] >= thr: add("고가구매형: 프리미엄/한정판, 무료 익일배송")
    if "PurchaseFrequency" in row and pd.notna(row["PurchaseFrequency"]) and "PurchaseFrequency" in ref_df.columns:
        thr = qv(ref_df["PurchaseFrequency"], 85)
        if thr is not None and row["PurchaseFrequency"] >= thr: add("자주구매형: 멤버십 상향, 묶음할인")
    if "TotalEngagementScore" in row and pd.notna(row["TotalEngagementScore"]) and "TotalEngagementScore" in ref_df.columns:
        thr = qv(ref_df["TotalEngagementScore"], 80)
        if thr is not None and row["TotalEngagementScore"] >= thr: add("참여형: 얼리액세스, 리뷰 리워드")
    if "EmailEngagementRate" in row and pd.notna(row["EmailEngagementRate"]) and "EmailEngagementRate" in ref_df.columns:
        thr = qv(ref_df["EmailEngagementRate"], 70)
        if thr is not None and row["EmailEngagementRate"] >= thr: add("이메일반응형: 개인화 쿠폰")
    if "MobileAppUsage" in row and pd.notna(row["MobileAppUsage"]) and "MobileAppUsage" in ref_df.columns:
        thr = qv(ref_df["MobileAppUsage"], 30)
        if thr is not None and row["MobileAppUsage"] < thr: add("앱저활성: 앱 온보딩/푸시 리마인드")
    if "AvgPurchaseInterval" in row and pd.notna(row["AvgPurchaseInterval"]) and "AvgPurchaseInterval" in ref_df.columns:
        thr = qv(ref_df["AvgPurchaseInterval"], 80)
        if thr is not None and row["AvgPurchaseInterval"] >= thr: add("구매주기긴형: 재구매 리마인드")
    if not tags:
        add("기본: VIP 전용 상담·무료반품·생일쿠폰")
    return " / ".join(tags)

# ---------------------------------------------------------------------
# 설정 영역(접기/펼치기) — 화면 구성 유지
# ---------------------------------------------------------------------
with st.expander("VIP 정의", expanded=False):
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        clv_q = st.slider("CLV 기준 분위(상위 %)", min_value=70, max_value=99, value=90, step=1)
    with colB:
        pf_q  = st.slider("구매빈도 기준 분위(상위 %)", min_value=60, max_value=95, value=80, step=1)
    with colC:
        logic = st.radio("VIP 판정 방식", ["AND (둘 다 충족)", "OR (둘 중 하나 충족)"], index=0, horizontal=True)

with st.expander("후보 선정 방식", expanded=False):
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        mode = st.selectbox("선정 모드", ["threshold(임계값)", "topk(상위 N)"], index=0)
    with col2:
        thr = st.slider("임계값 (VIP잠재지수)", 0, 100, 80, 1)
    with col3:
        topk = st.slider("상위 N (topk 모드)", 10, 1000, 100, 10)
    with col4:
        coverage_min_n = st.slider("최소 지표 수", 1, 6, 3, 1)
    col5, col6 = st.columns([1,1])
    with col5:
        strong_signal_pct = st.slider("강한 단일 신호 기준(상위 %)", 80, 99, 95, 1)
    with col6:
        include_nan_id_in_stats = st.checkbox("NaN ID도 통계에 포함(목록/CSV 제외)", value=False)

# ---------------------------------------------------------------------
# VIP 정의 계산(현재 VIP)
# ---------------------------------------------------------------------
clv_cut = qv(filtered["CustomerLifetimeValue"], clv_q) if "CustomerLifetimeValue" in filtered.columns else None
pf_cut  = qv(filtered["PurchaseFrequency"], pf_q) if "PurchaseFrequency" in filtered.columns else None
mask_clv = filtered["CustomerLifetimeValue"] >= (clv_cut if clv_cut is not None else -np.inf) if "CustomerLifetimeValue" in filtered.columns else False
mask_pf  = filtered["PurchaseFrequency"]   >= (pf_cut  if pf_cut  is not None else -np.inf) if "PurchaseFrequency" in filtered.columns else False
vip_mask = (mask_clv & mask_pf) if str(logic).startswith("AND") else (mask_clv | mask_pf)
vip_df = filtered[vip_mask].copy()

# 운영 원칙: 현재 VIP 표에서도 NaN ID 제외(링크/CSV 무의미)
if "CustomerID_clean" in vip_df.columns:
    vip_df = vip_df[vip_df["CustomerID_clean"].notna()]

# ---------------------------------------------------------------------
# 후보 스코어링 + 선정 (utils_vip 사용)
# ---------------------------------------------------------------------
scored = compute_vip_propensity_score(filtered, ref_df=filtered)
scored_full = filtered.reset_index(drop=True).merge(
    scored.reset_index(drop=True), left_index=True, right_index=True, how="left"
)
cands, snap = select_vip_candidates(
    scored_full,
    mode=("topk" if mode.startswith("topk") else "threshold"),
    k=int(topk), thr=float(thr),
    coverage_min_n=int(coverage_min_n),
    strong_signal_pct=float(strong_signal_pct),
    include_nan_id_in_stats=bool(include_nan_id_in_stats),
)

# ---------------------------------------------------------------------
# 탭 구성 — 화면 구성 유지
# ---------------------------------------------------------------------
tabs = st.tabs(["📌 개요", "🚀 전환 후보", "👑 현재 VIP", "ℹ️ 사용 설명"])

# == 개요 탭 ==
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전환 후보 수", f"{len(cands):,}")
    col2.metric("현재 VIP 수", f"{len(vip_df):,}")
    bt = backtest_metrics(scored_full, score_col="VIP잠재지수", label_col=None,
                          k=min(100, max(1, len(scored_full)//20)))
    col3.metric("Precision@K(프락시)", f"{bt['precision_at_k']*100:.1f}%")
    col4.metric("Lift@K(프락시)", f"{bt['lift_at_k']:.2f}x")

    roi = roi_for_k(scored_full, k=min(100, len(scored_full)),
                    avg_order_value=50000, gross_margin=0.35, cost_per_contact=1000)
    st.caption(
        f"예산 상위 {min(100, len(scored_full))}명 예상 EV ≈ ₩{roi['ev_total']:,.0f} "
        f"(1인당 기대가치 ₩{roi['ev_per_head']:,.0f}, 응답률≈{roi['p']*100:.1f}%)"
    )

    st.markdown("---")
    st.markdown(
        f"**선정 모드:** `{ 'topk' if mode.startswith('topk') else 'threshold' }`  "
        f"· **임계값/상위N:** `{topk if mode.startswith('topk') else thr}`  "
        f"· **최소 지표 수:** `{coverage_min_n}`  · **강한 단일 신호:** 상위 `{strong_signal_pct}%`"
    )

    with st.expander("🧮 전환 전략 시뮬레이터(후보 기준)"):
        base_aov = float(pd.to_numeric(filtered.get("AverageOrderValue", pd.Series([0])), errors="coerce").mean() or 0)
        base_pf  = float(pd.to_numeric(filtered.get("PurchaseFrequency", pd.Series([0])), errors="coerce").mean() or 0)
        cc1, cc2, cc3, cc4 = st.columns(4)
        conv_rate = cc1.slider("전환율(%)", 1, 100, 20, 1)
        lift_aov  = cc2.slider("AOV 상승(%)", 0, 100, 10, 1)
        lift_pf   = cc3.slider("구매빈도 상승(%)", 0, 100, 10, 1)
        cost_unit = cc4.number_input("인센티브 비용(원)", min_value=0, value=3000, step=500)
        n_target  = len(cands)

        add_rev_aov = base_aov * (lift_aov/100.0)
        add_rev_pf  = base_pf  * (lift_pf /100.0)
        add_per     = (add_rev_aov + add_rev_pf*base_aov) * (conv_rate/100.0)
        gross       = add_per * n_target
        cost        = cost_unit * n_target
        roi_val     = (gross - cost) / cost * 100 if cost>0 else 0.0

        cA,cB,cC = st.columns(3)
        cA.metric("추정 추가 매출(원)", f"{gross:,.0f}")
        cB.metric("추정 비용(원)", f"{cost:,.0f}")
        cC.metric("ROI(%)", f"{roi_val:,.1f}")

# ================================
# == 전환 후보 탭 (교체된 블록) ==
# ================================
with tabs[1]:
    st.subheader("🚀 전환 후보 리스트")
    if len(cands) == 0:
        st.info("현재 기준에서 후보가 없습니다. 임계값을 낮추거나 최소 지표 수를 완화해 보세요.")
    else:
        table_css()
        view = cands.copy()

        # (안전) ID 보강 + 표/CSV에서는 ID 없는 행 제외
        if "CustomerID_clean" not in view.columns and "CustomerID" in view.columns:
            tmp = view["CustomerID"].astype(str).str.strip()
            tmp = tmp.mask(tmp.str.lower().isin(["", "nan", "none", "null"]))
            view["CustomerID_clean"] = tmp
        if "CustomerID_clean" in view.columns:
            view = view[view["CustomerID_clean"].notna()].copy()

        # 고객ID 링크
        if "CustomerID_clean" in view.columns:
            view["고객ID"] = view["CustomerID_clean"].apply(
                lambda cid: f"<a href='/Customer_Detail?customer_id={quote(str(cid))}' target='_self'>{cid}</a>"
            )
            view.drop(columns=["CustomerID_clean"], inplace=True, errors="ignore")

        # 신뢰도 배지 + 추천 혜택
        cov = pd.to_numeric(view.get("coverage", 0.0), errors="coerce").fillna(0.0)
        def _badge(v):
            v = float(v)
            if v >= 0.75: return "<span class='badge green'>신뢰도 높음</span>"
            if v >= 0.45: return "<span class='badge'>신뢰도 보통</span>"
            return "<span class='badge'>신뢰도 낮음</span>"
        view["신뢰도"] = cov.apply(_badge)
        view["추천전략"] = [recommend_tags(row, filtered) for _, row in view.iterrows()]

        # VIP 게이지(막대)
        def bar_html(x):
            try:
                pct = int(np.clip(float(x), 0, 100))
            except Exception:
                pct = 0
            return f"<div class='barwrap'><div class='bar'><div class='fill' style='width:{pct}%;'></div></div><span>{pct}%</span></div>"
        if "VIP잠재지수" in view.columns:
            view["VIP게이지"] = view["VIP잠재지수"].apply(bar_html)

        # 한글 라벨로 변환 후 안전하게 컬럼 선택
        view = rename_for_display(view)
        metric_cols = [
            "PurchaseFrequency", "AverageOrderValue", "TotalEngagementScore",
            "EmailEngagementRate", "MobileAppUsage", "AvgPurchaseInterval",
            "NegativeExperienceIndex", "CSFrequency", "AverageSatisfactionScore"
        ]
        base_cols = [
            "고객ID", "VIP잠재지수", "VIP게이지", "신뢰도", "coverage",
            *metric_cols,
            "추천전략", "근거요약"
        ]
        display_cols = safe_cols(view, base_cols)

        # 포맷(한글 컬럼명 기준)
        fmt = {
            dlabel("AverageOrderValue"): "{:,.0f}",
            dlabel("PurchaseFrequency"): "{:.2f}",
            dlabel("TotalEngagementScore"): "{:.2f}",
            dlabel("EmailEngagementRate"): "{:.2f}",
            dlabel("MobileAppUsage"): "{:.0f}",
            dlabel("AvgPurchaseInterval"): "{:.2f}",
            dlabel("coverage"): "{:.2f}",
        }

        styler = view[display_cols].style.hide(axis="index").format(fmt)
        st.markdown(styler.set_table_attributes('id="pot_table"').to_html(escape=False), unsafe_allow_html=True)

        # 후보 표 전용 CSS(고객ID 열 너비 확보)
        st.markdown("""
        <style>
        #pot_table th:nth-child(1), #pot_table td:nth-child(1) { min-width: 120px; }
        </style>
        """, unsafe_allow_html=True)

        # CSV (고객ID 텍스트 포함, 게이지 제외)
        exp = view[display_cols].copy()
        if "고객ID" in exp.columns and "CustomerID" not in exp.columns:
            exp.insert(0, "CustomerID", exp["고객ID"].str.extract(r'>(.*?)<')[0])
            exp.drop(columns=["고객ID"], inplace=True)
        if "VIP게이지" in exp.columns:
            exp.drop(columns=["VIP게이지"], inplace=True)
        st.download_button("⬇️ 전환 후보 CSV", exp.to_csv(index=False).encode("utf-8-sig"),
                           "vip_candidates.csv", "text/csv")

# =============================
# == 현재 VIP 탭 (교체된 블록) ==
# =============================
with tabs[2]:
    st.subheader("👑 현재 VIP 고객")
    if len(vip_df) == 0:
        st.info("현재 VIP 고객이 없습니다. 상단 ‘VIP 정의’를 조정해 보세요.")
    else:
        table_css()
        view = vip_df.copy()

        # (안전) ID 보강 + 표/CSV에서는 ID 없는 행 제외
        if "CustomerID_clean" not in view.columns and "CustomerID" in view.columns:
            tmp = view["CustomerID"].astype(str).str.strip()
            tmp = tmp.mask(tmp.str.lower().isin(["", "nan", "none", "null"]))
            view["CustomerID_clean"] = tmp
        if "CustomerID_clean" in view.columns:
            view = view[view["CustomerID_clean"].notna()].copy()

        # 고객ID 링크
        if "CustomerID_clean" in view.columns:
            view["고객ID"] = view["CustomerID_clean"].apply(
                lambda cid: f"<a href='/Customer_Detail?customer_id={quote(str(cid))}' target='_self'>{cid}</a>"
            )
            view.drop(columns=["CustomerID_clean"], inplace=True, errors="ignore")

        # 추천 혜택
        view["추천혜택"] = [recommend_tags(row, filtered) for _, row in view.iterrows()]

        # 한글 라벨 후 안전 선택
        view = rename_for_display(view)
        base_cols = [
            "고객ID",
            "CustomerLifetimeValue", "PurchaseFrequency", "AverageOrderValue",
            "TotalEngagementScore", "EmailEngagementRate", "MobileAppUsage",
            "추천혜택"
        ]
        display_cols = safe_cols(view, base_cols)

        fmt = {
            dlabel("CustomerLifetimeValue"): "{:,.0f}",
            dlabel("PurchaseFrequency"): "{:.2f}",
            dlabel("AverageOrderValue"): "{:,.0f}",
            dlabel("TotalEngagementScore"): "{:.2f}",
            dlabel("EmailEngagementRate"): "{:.2f}",
            dlabel("MobileAppUsage"): "{:.0f}",
        }

        styler = view[display_cols].style.hide(axis="index").format(fmt)
        st.markdown(styler.set_table_attributes('id="vip_table"').to_html(escape=False), unsafe_allow_html=True)

        # 현재 VIP 표 전용 CSS(고객ID 열 너비 확보)
        st.markdown("""
        <style>
        #vip_table th:nth-child(1), #vip_table td:nth-child(1) { min-width: 120px; }
        </style>
        """, unsafe_allow_html=True)

        # CSV
        exp = view[display_cols].copy()
        if "고객ID" in exp.columns and "CustomerID" not in exp.columns:
            exp.insert(0, "CustomerID", exp["고객ID"].str.extract(r'>(.*?)<')[0])
            exp.drop(columns=["고객ID"], inplace=True)
        st.download_button("⬇️ VIP 리스트 CSV", exp.to_csv(index=False).encode("utf-8-sig"),
                           "vip_list.csv", "text/csv")

# == 사용 설명 탭 ==
with tabs[3]:
    st.subheader("ℹ️ 사용 설명")
    st.markdown("""
- **VIP 정의(🎯)**: CLV와 구매빈도를 기준으로 현재 VIP를 판정합니다.  
  - 분위(상위 %)와 AND/OR 논리를 조절해 VIP 범위를 정합니다.
- **후보 선정(🧪)**: `VIP잠재지수`는 **있는 지표만 정규화·가중합**하여 계산하고,  
  `coverage(데이터충분도)`로 **신뢰도 보정**이 적용됩니다.
  - 최소 지표 수를 만족하거나 **강한 단일 신호**(예: 평균주문금액/구매빈도 상위 95%)가 있으면 후보로 인정.
  - 최종 점수 = raw × (0.5 + 0.5 × √coverage) → **0~100점**.
- **NaN 처리(운영 원칙)**  
  1) `CustomerID`가 NaN이면 **리스트/CSV/링크에서 제외**(필요 시 통계에는 포함 가능).  
  2) 핵심지표 NaN은 **있는 지표만**으로 계산하며, coverage로 **자연 감점**.
- **추천 혜택**: 고객 패턴(고가구매/자주구매/참여형/앱저활성/재구매지연)에 맞춘 **전환 액션**을 제공합니다.
- **KPI(라벨 없는 환경용 프락시)**: Precision@K, Lift@K, 예상 ROI를 참고 지표로 제시합니다.
""")

    # ── 전략 시뮬레이터 안내(도움말 탭에 포함)
    with st.expander("🧮 전환 전략 시뮬레이터 안내", expanded=False):
        st.markdown("""
**무엇을 계산하나요?**  
현재 화면에서 선정된 **전환 후보 전체**를 대상으로 캠페인 집행 시 **추가 매출·비용·ROI(투자 대비 효과)** 를 빠르게 가정 계산합니다.

**어디에 있나요?**  
`📌 개요` 탭 하단의 **전환 전략 시뮬레이터(후보 기준)** 영역.

**입력값(관리자가 조정)**  
- **전환율(%)**: 연락한 후보 중 실제로 반응하는 비율  
- **평균주문금액 상승(%)**: 캠페인 후 AOV 상승율  
- **구매빈도 상승(%)**: 캠페인 후 구매 횟수 상승율  
- **인센티브 비용(원)**: 1인당 쿠폰/캐시백/사은품 비용

※ 기준이 되는 **현재 평균주문금액, 구매빈도**는 데이터에서 자동 추출됩니다.

**산출값(자동 표시)**  
- **추정 추가 매출(원)**  
- **추정 비용(원)**  
- **ROI(%) = (추정 추가 매출 − 추정 비용) ÷ 추정 비용 × 100**

**간단 산식**  
- 추정 추가 매출 ≈ 후보 수 × 전환율 × { (평균주문금액 × 상승%) + (구매빈도 × 상승% × 평균주문금액) }  
- 추정 비용 = 후보 수 × 인센티브 비용

**해석 팁**  
- 전환율·상승율을 높이면 **추정 추가 매출**↑, 인센티브 비용이 크면 **ROI**↓.  
- 후보 선정 조건(임계값/상위 N·최소 지표 수·강한 단일 신호)에 따라 **대상 수**가 변하므로  
  **후보 조건을 정한 뒤 → 시뮬레이터**를 조정하세요.

**유의사항**  
- 빠른 가정 계산 도구입니다. 실제 성과는 **응답률·마진·중복 접촉** 등 운영 요인에 좌우됩니다.  
- 필요 시 **마진율 반영, 세그먼트별 전환율 분리** 등으로 쉽게 고도화할 수 있습니다.
""")