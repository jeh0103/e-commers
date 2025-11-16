# pages/03_VIP_Insights.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os, json
from urllib.parse import quote

st.set_page_config(page_title="👑 VIP 인사이트", layout="wide")

# ===== 화면 라벨 =====
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
    "NegativeExperienceIndex": "부정경험지수",
    "CSFrequency": "상담빈도",
    "Age": "나이",
    "AnnualIncome": "연소득",
    "Income": "연소득",
    # 기존 모델 라벨(표시용)
    "IF_AnomalyScore": "패턴이탈지수(IF)",
    "AE_ReconError": "정상패턴차이(AE)",
}
def dlabel(c): return KOR_COL.get(c, c)
def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: dlabel(c) for c in df.columns})

# ===== 성별 라벨 보장(대시보드와 동일) =====
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

# ===== 데이터 로딩 =====
@st.cache_data(show_spinner=False)
def load_data():
    # 기본: 하이브리드 파일
    base = pd.read_csv("ecommerce_customer_churn_hybrid_with_id.csv")
    # CustomerID_clean
    if "CustomerID" in base.columns:
        def _clean_id(x):
            if pd.isna(x): return np.nan
            s = str(x).strip()
            return np.nan if (s=="" or s.lower() in {"nan","none","nat","null"}) else s
        base["CustomerID_clean"] = base["CustomerID"].map(_clean_id)

    # 성별 라벨
    base = ensure_gender_label(base)

    # 추가 피처가 있는 경우 조인
    if os.path.exists("ecommerce_customer_data_featured.csv"):
        feat = pd.read_csv("ecommerce_customer_data_featured.csv")
        if "CustomerID" in feat.columns:
            # 동일한 클린 ID 생성 후 조인
            def _clean_id2(x):
                if pd.isna(x): return np.nan
                s = str(x).strip()
                return np.nan if (s=="" or s.lower() in {"nan","none","nat","null"}) else s
            feat["CustomerID_clean"] = feat["CustomerID"].map(_clean_id2)
            # 덮어쓰지 않도록 필요한 컬럼만
            keep_cols = [c for c in feat.columns if c not in base.columns or c in
                         ["CustomerID","CustomerID_clean","CustomerLifetimeValue","AverageOrderValue",
                          "TotalPurchases","AvgPurchaseInterval","EmailEngagementRate","MobileAppUsage",
                          "TotalEngagementScore","AnnualIncome","Income"]]
            base = base.merge(feat[keep_cols], on=["CustomerID","CustomerID_clean"], how="left")
    return base

df = load_data()

# ===== 전역 필터(대시보드 공유) 적용 =====
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

# ===== 필수 컬럼 체크 =====
need_cols = ["CustomerID_clean","CustomerLifetimeValue","PurchaseFrequency"]
missing = [c for c in need_cols if c not in filtered.columns]
if missing:
    st.error("VIP 분석에 필요한 컬럼이 없습니다: " + ", ".join([dlabel(c) for c in missing]))
    st.stop()

# ===== 페이지 헤더 =====
st.title("👑 VIP 인사이트")
st.caption("CLV·구매빈도 기반 VIP 정의와 VIP 성장 가능성 예측, 공통 특성/권장 액션을 제공합니다.")

# ===== 기준 설정 =====
with st.expander("⚙️ VIP 기준 설정", expanded=True):
    clv_q = st.slider("CLV 기준 분위(상위 %)", min_value=70, max_value=99, value=90, step=1)
    pf_q  = st.slider("구매빈도 기준 분위(상위 %)", min_value=60, max_value=95, value=80, step=1)
    logic = st.radio("VIP 판정 방식", ["AND (둘 다 충족)", "OR (둘 중 하나 충족)"], index=0, horizontal=True)
with st.expander("⚙️ 성장 가능 VIP 기준", expanded=False):
    pot_thr = st.slider("VIP 잠재지수 임계(0~100)", min_value=50, max_value=99, value=80, step=1)

# 분위수 값 계산
def qv(s, q): 
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q/100.0)) if len(s)>0 else None

clv_cut = qv(filtered["CustomerLifetimeValue"], clv_q)
pf_cut  = qv(filtered["PurchaseFrequency"], pf_q)

mask_clv = filtered["CustomerLifetimeValue"] >= (clv_cut if clv_cut is not None else -np.inf)
mask_pf  = filtered["PurchaseFrequency"]   >= (pf_cut  if pf_cut  is not None else -np.inf)
vip_mask = (mask_clv & mask_pf) if logic.startswith("AND") else (mask_clv | mask_pf)

vip_df   = filtered[vip_mask].copy()
nonvip_df= filtered[~vip_mask].copy()

# ===== VIP 잠재지수(0-100) 산출 =====
def qnorm(ref, v, low=0.05, high=0.95, invert=False):
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    v   = pd.to_numeric(v, errors="coerce")
    if len(ref)==0:
        x = (v - v.min()) / max(1e-9, (v.max()-v.min()))
    else:
        lo, hi = float(ref.quantile(low)), float(ref.quantile(high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
            lo, hi = float(ref.min()), float(ref.max())
        x = (v - lo) / max(1e-9, (hi-lo))
    x = x.clip(0,1)
    if invert: x = 1 - x
    return (x * 100).round(0)

# 기본 규칙 기반 스코어(라이브러리 없이 동작)
def rule_based_propensity(df_ref, df_eval):
    comps = []
    def add(col, w=1.0, invert=False):
        if col in df_eval.columns:
            comps.append((qnorm(df_ref[col], df_eval[col], invert=invert), w, col))
    # 양의 기여
    add("PurchaseFrequency", 0.22)
    add("AverageOrderValue", 0.20)
    add("TotalPurchases", 0.15)
    add("TotalEngagementScore", 0.15)
    add("EmailEngagementRate", 0.08)
    add("MobileAppUsage", 0.08)
    # 음의 기여(작을수록 좋음)
    add("AvgPurchaseInterval", 0.06, invert=True)
    add("NegativeExperienceIndex", 0.04, invert=True)
    add("CSFrequency", 0.02, invert=True)
    if not comps:
        return pd.Series([0]*len(df_eval), index=df_eval.index, dtype=float)
    parts = [w*vec for (vec,w,_) in comps]
    total_w = sum(w for (_,w,_) in comps)
    score = sum(parts) / total_w
    return score.round(0)

# (선택) scikit-learn로 로지스틱 회귀 학습 → 가능하면 사용, 아니면 규칙 기반
def model_based_propensity(df_ref, df_eval, vip_mask_ref):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        use_cols = [c for c in [
            "PurchaseFrequency","AverageOrderValue","TotalPurchases","TotalEngagementScore",
            "EmailEngagementRate","MobileAppUsage","AvgPurchaseInterval",
            "NegativeExperienceIndex","CSFrequency","AverageSatisfactionScore","Age"
        ] if c in df_ref.columns]
        if not use_cols: 
            return rule_based_propensity(df_ref, df_eval)
        X = df_ref[use_cols].fillna(df_ref[use_cols].median(numeric_only=True))
        y = vip_mask_ref.astype(int).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xs, y)
        Xe = df_eval[use_cols].fillna(df_ref[use_cols].median(numeric_only=True))
        Xes= scaler.transform(Xe)
        proba = clf.predict_proba(Xes)[:,1]
        return pd.Series((proba*100).round(0), index=df_eval.index)
    except Exception:
        return rule_based_propensity(df_ref, df_eval)

# 잠재지수 계산 (비VIP만 대상)
pot_df = nonvip_df.copy()
pot_df["VIP잠재지수"] = model_based_propensity(filtered, pot_df, vip_mask)

# 임계 이상 후보
pot_cand = pot_df[pot_df["VIP잠재지수"] >= pot_thr].copy()

# ===== 스냅샷 =====
col1, col2, col3 = st.columns(3)
col1.metric("현재 VIP 수", f"{len(vip_df):,}")
col2.metric("성장 가능 VIP 후보 수", f"{len(pot_cand):,}")
if "CustomerLifetimeValue" in filtered.columns:
    avg_v, avg_all = float(vip_df["CustomerLifetimeValue"].mean()) if len(vip_df)>0 else 0.0,\
                     float(filtered["CustomerLifetimeValue"].mean())
    delta = ( (avg_v-avg_all)/avg_all*100 if avg_all>0 else 0.0 )
    col3.metric("VIP 평균 CLV", f"{avg_v:,.0f}", f"{delta:+.1f}% vs 전체")

# ===== VIP 공통 특성 요약 =====
def profile_diff(vip, nonvip, col):
    v = pd.to_numeric(vip[col], errors="coerce"); n = pd.to_numeric(nonvip[col], errors="coerce")
    if v.notna().sum()<5 or n.notna().sum()<5: return None
    dv = float(v.median()); dn = float(n.median())
    base = float(pd.to_numeric(filtered[col], errors="coerce").median())
    if not np.isfinite(base) or base==0: return None
    return (dv-dn)/abs(base)

key_cols = [c for c in [
    "AverageOrderValue","PurchaseFrequency","TotalPurchases","TotalEngagementScore",
    "EmailEngagementRate","MobileAppUsage","AvgPurchaseInterval",
    "AverageSatisfactionScore","NegativeExperienceIndex","CSFrequency","AnnualIncome","Income"
] if c in filtered.columns]

bullets = []
for c in key_cols:
    diff = profile_diff(vip_df, nonvip_df, c)
    if diff is None: continue
    direction = "↑" if diff>0 else "↓"
    txt = f"- **{dlabel(c)} {direction}** (VIP vs 비VIP 상대차: {diff*100:+.1f}%)"
    bullets.append(txt)

st.subheader("🧭 VIP 공통 특성(요약)")
if bullets:
    st.markdown("\n".join(bullets))
else:
    st.info("비교 가능한 지표가 충분하지 않습니다.")

st.markdown("---")

# ===== 추천 혜택/액션 태그 생성 =====
def recommend_tags(row):
    tags = []
    def add(t): tags.append(t)
    # 유형 감지
    if "AverageOrderValue" in row and pd.notna(row["AverageOrderValue"]) and row["AverageOrderValue"]>= qv(filtered["AverageOrderValue"], 85):
        add("고가구매형: 프리미엄/한정판 우선구매, 무료 익일배송")
    if "PurchaseFrequency" in row and pd.notna(row["PurchaseFrequency"]) and row["PurchaseFrequency"]>= qv(filtered["PurchaseFrequency"], 85):
        add("자주구매형: 멤버십 등급상향, 묶음할인")
    if "TotalEngagementScore" in row and pd.notna(row["TotalEngagementScore"]) and row["TotalEngagementScore"]>= qv(filtered["TotalEngagementScore"], 80):
        add("참여형: 얼리액세스, 리뷰 리워드")
    if "EmailEngagementRate" in row and pd.notna(row["EmailEngagementRate"]) and row["EmailEngagementRate"]>= qv(filtered["EmailEngagementRate"], 70):
        add("이메일반응형: 개인화 쿠폰·맞춤 카탈로그")
    if "MobileAppUsage" in row and pd.notna(row["MobileAppUsage"]) and row["MobileAppUsage"]< qv(filtered["MobileAppUsage"], 30):
        add("앱저활성: 앱 첫구매 추가혜택·푸시 온보딩")
    if "AvgPurchaseInterval" in row and pd.notna(row["AvgPurchaseInterval"]) and row["AvgPurchaseInterval"]>= qv(filtered["AvgPurchaseInterval"], 80):
        add("구매주기긴형: 리마인드·재구매 쿠폰")
    if not tags:
        add("기본: VIP 전용 상담·무료반품·생일쿠폰")
    return " / ".join(tags)

# ===== 표 스타일 공통 =====
def table_css():
    st.markdown("""
    <style>
    #vip_table, #pot_table { width: 100% !important; table-layout: fixed; }
    #vip_table th, #vip_table td, #pot_table th, #pot_table td {
      padding: 10px 12px !important; line-height: 1.45; vertical-align: middle;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .barwrap { display:flex; align-items:center; gap:8px; }
    .bar    { flex:1; height:10px; background:rgba(0,0,0,0.08); border-radius:999px; overflow:hidden; }
    .bar .fill { height:100%; background: rgba(0, 122, 255, 0.55); }
    .badge { padding: 2px 6px; border-radius: 6px; font-size: 12px; line-height: 1;
             background: rgba(0,0,0,0.06); }
    .badge.gold { background: rgba(255,204,0,.18); border: 1px solid rgba(255,204,0,.35); }
    .badge.green{ background: rgba(52,199,89,.18); border: 1px solid rgba(52,199,89,.35); }
    </style>
    """, unsafe_allow_html=True)

def bar_html(x, x_max):
    try:
        pct = int(np.clip(100.0 * float(x)/max(1e-9, float(x_max)), 0, 100))
    except Exception:
        pct = 0
    return f"<div class='barwrap'><div class='bar'><div class='fill' style='width:{pct}%;'></div></div><span>{pct}%</span></div>"

# ===== (A) 현재 VIP 리스트 =====
st.subheader("👑 현재 VIP 고객")
if len(vip_df)==0:
    st.info("현재 VIP 고객이 없습니다. 기준을 완화해 보세요.")
else:
    table_css()
    show_cols = [c for c in [
        "CustomerID_clean","GenderLabel","CustomerLifetimeValue","PurchaseFrequency",
        "AverageOrderValue","TotalEngagementScore","EmailEngagementRate","MobileAppUsage"
    ] if c in vip_df.columns]
    view = vip_df[show_cols].copy()
    view.insert(0, "", np.arange(1, len(view) + 1))
    # 링크
    view["고객ID"] = view["CustomerID_clean"].apply(lambda cid: f"<a href='/Customer_Detail?customer_id={quote(str(cid))}' target='_self'>{cid}</a>")
    # 시각화 컬럼
    aov_max = float(filtered["AverageOrderValue"].max()) if "AverageOrderValue" in filtered.columns else 1.0
    pf_max  = float(filtered["PurchaseFrequency"].max()) if "PurchaseFrequency" in filtered.columns else 1.0
    if "AverageOrderValue" in view.columns: view["AOV시각화"] = view["AverageOrderValue"].apply(lambda v: bar_html(v, aov_max))
    if "PurchaseFrequency"  in view.columns: view["구매빈도시각화"] = view["PurchaseFrequency"].apply(lambda v: bar_html(v, pf_max))
    # 추천 태그
    view["추천혜택"] = [recommend_tags(row) for _, row in view.iterrows()]
    # 정리
    view.drop(columns=["CustomerID_clean"], inplace=True, errors="ignore")
    view = rename_for_display(view)
    # 컬럼 순서
    order = ["", "고객ID", "고객생애가치(CLV)", "구매빈도", "AOV시각화", "구매빈도시각화", "추천혜택"]
    order += [c for c in view.columns if c not in order]
    styler = view[order].style.hide(axis="index").format({dlabel("CustomerLifetimeValue"): "{:,.0f}",
                                                          dlabel("PurchaseFrequency"): "{:.2f}",
                                                          dlabel("AverageOrderValue"): "{:,.0f}"})
    st.markdown(styler.set_table_attributes('id="vip_table"').to_html(escape=False), unsafe_allow_html=True)
    # CSV
    exp = view.copy()
    exp.rename(columns={"": "순위"}, inplace=True)
    exp.insert(1, "CustomerID", exp["고객ID"].str.extract(r'>(.*?)<')[0])
    exp.drop(columns=["고객ID","AOV시각화","구매빈도시각화"], inplace=True, errors="ignore")
    st.download_button("⬇️ VIP 리스트 CSV", exp.to_csv(index=False).encode("utf-8-sig"), "vip_list.csv", "text/csv")

st.markdown("---")

# ===== (B) 성장 가능 VIP 후보 =====
st.subheader("🚀 VIP 성장 가능 고객")
if len(pot_cand)==0:
    st.info("현재 기준의 잠재지수 임계 이상 후보가 없습니다. 임계를 낮추거나 전역 필터를 완화해 보세요.")
else:
    table_css()
    show_cols2 = [c for c in [
        "CustomerID_clean","GenderLabel","VIP잠재지수","PurchaseFrequency","AverageOrderValue",
        "TotalEngagementScore","EmailEngagementRate","MobileAppUsage","AvgPurchaseInterval",
        "NegativeExperienceIndex","CSFrequency"
    ] if c in pot_cand.columns]
    view2 = pot_cand[show_cols2].copy()
    view2.insert(0, "", np.arange(1, len(view2) + 1))
    view2["고객ID"] = view2["CustomerID_clean"].apply(lambda cid: f"<a href='/Customer_Detail?customer_id={quote(str(cid))}' target='_self'>{cid}</a>")
    # 등급 배지
    def tier(x):
        x = float(x) if pd.notna(x) else 0.0
        if x>=90: return "<span class='badge gold'>매우 높음</span>"
        if x>=75: return "<span class='badge green'>높음</span>"
        return "<span class='badge'>보통</span>"
    view2["등급"] = view2["VIP잠재지수"].apply(tier)
    # 추천액션
    view2["추천혜택"] = [recommend_tags(row) for _, row in view2.iterrows()]
    # 정리
    view2.drop(columns=["CustomerID_clean"], inplace=True, errors="ignore")
    view2 = rename_for_display(view2)
    order2 = ["", "고객ID", "VIP잠재지수", "등급", "추천혜택", dlabel("PurchaseFrequency"), dlabel("AverageOrderValue"),
              dlabel("TotalEngagementScore"), dlabel("EmailEngagementRate"), dlabel("MobileAppUsage")]
    order2 += [c for c in view2.columns if c not in order2]
    fmt2 = { "VIP잠재지수": "{:.0f}", dlabel("PurchaseFrequency"): "{:.2f}", dlabel("AverageOrderValue"): "{:,.0f}" }
    styler2 = view2[order2].style.hide(axis="index").format(fmt2)
    st.markdown(styler2.set_table_attributes('id="pot_table"').to_html(escape=False), unsafe_allow_html=True)
    # CSV
    exp2 = view2.copy()
    exp2.rename(columns={"": "순위"}, inplace=True)
    exp2.insert(1, "CustomerID", exp2["고객ID"].str.extract(r'>(.*?)<')[0])
    exp2.drop(columns=["고객ID"], inplace=True)
    st.download_button("⬇️ 성장 가능 VIP CSV", exp2.to_csv(index=False).encode("utf-8-sig"), "vip_potential.csv", "text/csv")